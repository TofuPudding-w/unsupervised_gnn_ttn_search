"""Independent artifact audit and trained-encoder sanity checks."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from ttn_search.evaluate import write_json
from ttn_search.learning.encoder import Encoder, batch, embed
from ttn_search.search import initial_ids
from ttn_search.trees import canonical, comb, tree_hash

OUT = Path('artifacts/stage_b')


def main():
    torch.set_num_threads(1)
    data = json.loads((OUT/'corpus.json').read_text())
    hashes = [{tree_hash(canonical(t)) for t in data[k]} for k in ['train','validation','test']]
    assert [len(s) for s in hashes]==[2000,250,250]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    all_hashes = set.union(*hashes)
    for p in Path('artifacts/stage_a_pilot_v2').glob('*/pool.json'):
        original = {tree_hash(canonical(json.loads(t))) for t in json.loads(p.read_text())['trees']}
        assert not all_hashes & original
    checked = 0
    minima = []
    for p in OUT.glob('*/records.json'):
        records = json.loads(p.read_text())
        pool = [canonical(t) for t in json.loads((p.parent/'pool.json').read_text())['trees']]
        assert len(records)==len(pool)==128
        for t,r in zip(pool,records):
            assert r['tree_hash']==tree_hash(t)
            minima.append(r['fidelity_final'])
        for trace_path in p.parent.glob('*.json'):
            if trace_path.stem in ['pool','records']: continue
            trace = json.loads(trace_path.read_text())
            seed = int(trace_path.stem.rsplit('_',1)[1])
            assert len(trace)==64
            assert [r['candidate_id'] for r in trace[:8]]==initial_ids(128,8,seed)
            seen = set(); best = None
            for r in trace:
                i = r['candidate_id']; assert i not in seen; seen.add(i)
                rec = records[i]; assert rec['evaluation_key']==r['evaluation_key']
                if rec['status']=='success' and rec['fidelity_final']>=.999:
                    best = rec['cx_count'] if best is None else min(best,rec['cx_count'])
                assert r['best_cx']==best
                checked += 1
    checks = []
    testing = [canonical(t) for t in data['test']]
    for seed in range(3):
        model = Encoder()
        model.load_state_dict(torch.load(OUT/f'gnn_{seed}.pt',weights_only=True))
        model.eval()
        x,a = batch([comb(list(range(8)))])
        torch.manual_seed(100)
        perm = torch.randperm(15)
        with torch.no_grad():
            z = model(x,a)
            error = float((z-model(x[:,perm],a[:,perm][:,:,perm])).abs().max())
            torch.testing.assert_close(z,model(x[:,perm],a[:,perm][:,:,perm]),atol=1e-5,rtol=1e-5)
            delta = float((z-model(*batch([comb(list(reversed(range(8))))]))).abs().max())
            assert delta>1e-6
        start=time.perf_counter(); latent=embed(model,testing); elapsed=time.perf_counter()-start
        singular=np.linalg.svd(latent-latent.mean(axis=0),compute_uv=False)
        assert np.isfinite(latent).all() and latent.std(axis=0).min()>1e-6
        checks.append(dict(seed=seed,node_permutation_max_error=error,comb_label_change=delta,
            latent_std=latent.std(axis=0).tolist(),centered_singular_values=singular.tolist(),
            encoding_250_trees_s=elapsed))
    write_json(OUT/'audit.json',dict(trace_steps_checked=checked,min_final_fidelity=min(minima),
        disjoint_corpus_and_candidate_pools=True,encoder_checks=checks))
    print('Audited',checked,'trace steps; min fidelity',min(minima))


if __name__=='__main__':main()
