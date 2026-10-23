"""Fixed-corpus 300-epoch follow-up; first 100 epochs must reproduce the pilot."""
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from decision_common import freeze,STAGE_B
from run_stage_b import replay
from ttn_search.evaluate import write_json
from ttn_search.learning.encoder import train,embed,batch,loss
from ttn_search.search import gp_predict,initial_ids
from ttn_search.trees import canonical,features


def main():
    torch.set_num_threads(1)
    config,out=freeze('gnn');started=time.perf_counter()
    data=json.loads((STAGE_B/'corpus.json').read_text())
    training,validation,testing=[[canonical(t) for t in data[k]] for k in ['train','validation','test']]
    old_training=json.loads((STAGE_B/'training.json').read_text())
    results=[];models={};scalers={}
    for seed in config['gnn_seeds']:
        model,result=train(training,validation,seed,epochs=config['gnn_epochs'])
        prior=next(r for r in old_training if r['seed']==seed)
        discrepancy=max(abs(result['history'][i][k]-prior['history'][i][k]) for i in range(100) for k in ['train_loss','validation_loss'])
        result['first_100_max_absolute_difference']=discrepancy
        assert discrepancy<1e-9, 'Controlled training failed to reproduce the first 100 epochs'
        with torch.no_grad():
            truth=torch.tensor(np.array([features(t) for t in testing]),dtype=torch.float32)
            result['test_loss']=float(loss(model.decoder(model(*batch(testing))),truth))
        result['old_test_loss']=prior['test_loss']
        result['old_best_validation_loss']=prior['best_validation_loss']
        torch.save(model.state_dict(),out/f'gnn_{seed}.pt')
        models[f'gnn_{seed}']=model
        scalers[f'gnn_{seed}']=StandardScaler().fit(embed(model,training))
        results.append(result);write_json(out/'training.json',results)
        print('trained',seed,'best epoch',result['best_epoch'],'test loss',result['test_loss'],'reproduction error',discrepancy,flush=True)
    predictions=[];trials=[]
    for family in ['hierarchical','overlapping']:
        for target_seed in range(3):
            label=f'{family}_{target_seed}'
            pool=[canonical(t) for t in json.loads((STAGE_B/label/'pool.json').read_text())['trees']]
            records=json.loads((STAGE_B/label/'records.json').read_text())
            costs=np.array([r['cx_count'] for r in records]); assert all(r['status']=='success' for r in records)
            for method,model in models.items():
                x=scalers[method].transform(embed(model,pool))
                for size in config['known_counts']:
                    for seed in config['split_seeds']:
                        known=initial_ids(len(pool),size,seed);held=[i for i in range(len(pool)) if i not in known]
                        mean,_,warnings=gp_predict(x,known,np.log1p(costs[known]),held)
                        chosen=np.asarray(held)[np.argsort(mean,kind='stable')[:5]]
                        rho=float(spearmanr(mean,costs[held])[0])
                        predictions.append(dict(target=label,family=family,method=method,train_size=size,seed=seed,
                            known=known,held=held,recommended=chosen.tolist(),spearman=rho if np.isfinite(rho) else None,
                            top5_regret=int(min(costs[chosen])-min(costs[held])),warnings=warnings))
                for seed in config['search_seeds']:
                    trace=replay(pool,records,x,method,seed,budget=64)
                    assert len({r['candidate_id'] for r in trace})==64
                    assert [r['candidate_id'] for r in trace[:8]]==initial_ids(128,8,seed)
                    best=None
                    for row in trace:
                        r=records[row['candidate_id']]
                        if r['status']=='success' and r['fidelity_final']>=.999: best=r['cx_count'] if best is None else min(best,r['cx_count'])
                        assert row['best_cx']==best
                    write_json(out/label/f'{method}_{seed}.json',trace)
                    trials.append(dict(target=label,family=family,seed=seed,method=method,
                        checkpoints={str(b):trace[b-1]['best_cx'] for b in [8,16,32,64]}))
            write_json(out/'prediction.json',predictions);write_json(out/'search.json',trials)
            print(label,'downstream checks complete',flush=True)
    write_json(out/'completion.json',dict(elapsed_s=time.perf_counter()-started,
        trajectories=len(trials),prediction_fits=len(predictions),training_s=sum(r['training_time_s'] for r in results)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
