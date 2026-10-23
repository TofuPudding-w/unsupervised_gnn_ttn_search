"""Independent checks of saved decision-experiment recommendations and accounting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ttn_search.evaluate import write_json

ROOT=Path('artifacts/decision_experiment')


def read(path):return json.loads(path.read_text())


def main():
    cost=ROOT/'cost';gnn=ROOT/'gnn'
    for component in [cost,gnn]:
        protocol=read(component/'protocol.json')
        for name,expected in protocol['script_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
    descriptor=read(cost/'descriptors.json');cohorts=read(cost/'cohorts.json')
    predictions=read(cost/'prediction.json')
    for row in predictions:
        keys=cohorts[row['cohort']+'/'+row['target']]
        costs=np.array([descriptor[k]['cx'] for k in keys])
        known=set(row['known']);held=set(row['held']);recommended=row['recommended']
        assert not known & held and known|held==set(range(len(keys)))
        assert len(known)==row['known_count'] and 0 in known
        assert len(set(recommended))==5 and set(recommended)<=held
        assert row['top5_regret']==min(costs[recommended])-min(costs[row['held']])
        if not row['method'].endswith('_gp'):
            index=read(cost/'protocol.json')['config']['descriptor_names'].index(row['method'])
            scores=np.array([descriptor[k]['features'][index] for k in keys])
            assert max(scores[recommended])<=sorted(scores[row['held']])[4]
    screening=read(cost/'screening.json')
    for row in screening:
        keys=cohorts[row['cohort']+'/'+row['target']]
        costs=np.array([descriptor[k]['cx'] for k in keys])
        order=np.random.default_rng(row['seed']).permutation(np.arange(1,len(keys)))
        if row['method']!='random':
            index=read(cost/'protocol.json')['config']['descriptor_names'].index(row['method'])
            scores=np.array([descriptor[k]['features'][index] for k in keys])
            order=order[np.argsort(scores[order],kind='stable')]
            assert abs(row['construction_s']-sum(descriptor[k]['construction_s'] for k in keys))<1e-9
        order=np.concatenate(([0],order))
        assert len(set(order))==len(keys)
        for budget,value in row['checkpoints'].items():assert value==min(costs[order[:int(budget)]])
    transition=read(cost/'transition.json')['rows']
    for r in transition:
        assert len(r['blocks'])==7
        assert r['feasible']
        assert sum(n['cx'] for n in r['blocks'])==r['cx']
        for n in r['blocks']:
            assert n['padded_columns']==2**n['input_bits'] and n['padded_columns']>=n['rank']
            assert n['padded_rows']==2**n['output_bits']
    result=dict(descriptor_records=len(descriptor),prediction_comparisons=len(predictions),
        static_screening_runs=len(screening),transition_variants=len(transition),gnn_complete=False)
    if (gnn/'completion.json').exists():
        old_training=read(Path('artifacts/stage_b/training.json'))
        for r in read(gnn/'training.json'):
            prior=next(a for a in old_training if a['seed']==r['seed'])
            assert len(r['history'])==300
            assert max(abs(r['history'][i][k]-prior['history'][i][k]) for i in range(100) for k in ['train_loss','validation_loss'])<1e-9
            assert r['best_epoch']==min(r['history'],key=lambda h:h['validation_loss'])['epoch']
        steps=0
        for summary in read(gnn/'search.json'):
            target=summary['target'];method=summary['method'];seed=summary['seed']
            trace=read(gnn/target/f'{method}_{seed}.json')
            old=read(Path('artifacts/stage_b')/target/f'{method}_{seed}.json')
            records=read(Path('artifacts/stage_b')/target/'records.json')
            assert [r['candidate_id'] for r in trace[:8]]==[r['candidate_id'] for r in old[:8]]
            assert len(trace)==64 and len({r['candidate_id'] for r in trace})==64
            best=None
            for r in trace:
                record=records[r['candidate_id']]
                assert r['evaluation_key']==record['evaluation_key']
                if record['status']=='success' and record['fidelity_final']>=.999:
                    best=record['cx_count'] if best is None else min(best,record['cx_count'])
                assert r['best_cx']==best
                steps+=1
            for b,value in summary['checkpoints'].items():assert trace[int(b)-1]['best_cx']==value
        for r in read(gnn/'prediction.json'):
            costs=np.array([a['cx_count'] for a in read(Path('artifacts/stage_b')/r['target']/'records.json')])
            assert set(r['recommended'])<=set(r['held'])
            assert not set(r['known']) & set(r['held'])
            assert r['top5_regret']==min(costs[r['recommended']])-min(costs[r['held']])
        result.update(gnn_complete=True,gnn_search_steps=steps)
    write_json(ROOT/'audit.json',result)
    print(result)


if __name__=='__main__':main()
