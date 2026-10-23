"""Offline completion of the frozen C1 overlapping pool; not an online benchmark."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline,source_hash,software,write_json
from ttn_search.screening import evaluator
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical

BASE=Path('artifacts/stage_c1')
OUT=Path('artifacts/stage_c1_headroom')

def read(p):return json.loads(Path(p).read_text())
def fingerprint(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(target,tree,directory):
    original=evaluator.compile_reference
    def compile_logged(*args,**kwargs):
        # Persist before entering synthesis so termination cannot hide an attempt.
        with (directory/'attempt_starts.jsonl').open('a') as f:
            f.write(json.dumps(dict(monotonic=time.monotonic()))+'\n');f.flush();os.fsync(f.fileno())
        return original(*args,**kwargs)
    evaluator.compile_reference=compile_logged
    try:
        with threadpool_limits(limits=1):
            evaluator.evaluate_fresh(target,tree,Pipeline(),directory)
    except Exception as exc:
        write_json(directory/'worker_error.json',dict(status='worker_error',error=repr(exc)))

def report(rows,score_rows):
    feasible=[r for r in rows if r['status']=='success']
    baseline=next(r['cx'] for r in feasible if r['candidate_id']==0)
    best=min(r['cx'] for r in feasible)
    improving=[r for r in feasible if r['cx']<baseline]
    statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    orders={}
    for key in ['parameters','padded']:
        values=np.array([r[key] if r[key] is not None else np.inf for r in score_rows])
        tie=np.random.default_rng(0).permutation(len(rows))
        orders[key]=tie[np.argsort(values[tie],kind='stable')].tolist()
    lines=['# C1 overlapping-pool headroom diagnostic','',
        '## Scope','',
        'Offline diagnostic of the original 10-qubit overlapping target, seed 13, and its frozen 128-tree pool. Existing completed records are reused; each remaining candidate receives one evaluation with the original internal precision retries and fidelity requirement 0.999. New evaluations have a 60-second per-candidate ceiling. No new candidate generator, score, target, or compiler is introduced.','',
        'Record selection is frozen before new compilation: the lexicographically first existing completed record for each tree is used, independent of its cost. All duplicate records are checked for status/cost agreement. Completed artifacts from deadline runs may be reused here even if they were not credited online; this does not alter the original benchmark.','',
        '## Headroom','',f'- Status counts: {statuses}.',f'- SCENT: **{baseline} CNOTs**.',
        f'- Best verified candidate: **{best} CNOTs**.',f'- SCENT gap to best verified: **{baseline-best} CNOTs ({100*(baseline-best)/baseline:.2f}%)**.',
        f'- Strict improvements over SCENT: **{len(improving)}/{len(rows)}**.',
        f'- Verified ties with SCENT: **{sum(r["cx"]==baseline for r in feasible)}**.','']
    unknown=[r for r in rows if r['status'] not in ['success','infeasible']]
    lines+=['All candidates have completed measured evaluations under the fixed policy; the best feasible value is the policy-defined pool optimum.' if not unknown else
        f'{len(unknown)} candidates remain unresolved due to operational failures/timeouts. The best verified value is not a certified pool optimum.','',
        '## Score retrieval','',
        'Ranks are one-based; ties use the original repetition-0 random permutation, without cost labels. Best-of-K is an offline replay, not a fresh timing experiment.','',
        '| Score | SCENT rank | First improvement rank | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 | K=128 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    lookup={r['candidate_id']:r for r in rows}
    for key,order in orders.items():
        first=next((i+1 for i,c in enumerate(order) if lookup[c]['status']=='success' and lookup[c]['cx']<baseline),None)
        vals=[]
        for k in [1,2,4,8,16,32,64,128]:
            costs=[lookup[i]['cx'] for i in order[:k] if lookup[i]['status']=='success'];vals.append(str(min(costs)) if costs else '—')
        lines.append(f'| {key} | {order.index(0)+1} | {first if first else "none"} | '+' | '.join(vals)+' |')
    lines+=['','## Improving candidates','', '| Candidate | CNOT | Gain over SCENT | Parameter rank | Padded rank |','|---|---:|---:|---:|---:|']
    for r in sorted(improving,key=lambda r:(r['cx'],r['candidate_id'])):
        i=r['candidate_id'];lines.append(f'| {i} | {r["cx"]} | {baseline-r["cx"]} | {orders["parameters"].index(i)+1} | {orders["padded"].index(i)+1} |')
    if not improving:lines+=['','No evaluated candidate strictly improves SCENT. This pool provides no observed improvement opportunity; it does not show whether better trees exist outside the pool.']
    lines+=['','## Cost and limitations','',
        f'Reused {sum(r["reused"] for r in rows)} records; newly evaluated {sum(not r["reused"] for r in rows)} candidates. New controlled evaluation wall time totals {sum(r.get("wall_s",0) for r in rows):.2f} seconds. New synthesis starts, including interrupted attempts: {sum(r.get("synthesis_starts",0) for r in rows)}.',
        '', 'New synthesis attempts are flushed and fsynced before compilation. JSON results use the existing atomic temporary-file replacement helper. Offline reuse and candidate time ceilings make these costs unsuitable as online speedup evidence. A single target cannot establish size-dependent behavior or compiler independence. Larger/mixed GNN pretraining remains on hold.','']
    write_json(OUT/'results.json',rows);(OUT/'REPORT.md').write_text('\n'.join(lines))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pool=read(BASE/'overlapping/pool.json');target=gaussian_target(10,'overlapping',13)
    old=read(BASE/'protocol.json')
    assert source_hash()==old['core_hash'] and software()==old['software']
    assert target.hash==pool['target_hash']
    candidates={}
    for p in sorted((BASE/'overlapping').glob('*/*/candidate_*/record.json')):
        rec=read(p);assert rec['target_hash']==target.hash
        candidates.setdefault(rec['tree_hash'],[]).append((p,rec))
    selected={}; disagreements=[]
    for key,items in candidates.items():
        selected[key]=str(items[0][0])
        if len({(r['status'],r.get('cx_count')) for _,r in items})>1:disagreements.append(key)
    assert not disagreements, 'Duplicate outcomes differ; investigate before pooling'
    protocol=dict(core_hash=source_hash(),software=software(),runner_hash=fingerprint(__file__),
        evaluator_hash=fingerprint('src/ttn_search/screening/evaluator.py'),pool_hash=fingerprint(BASE/'overlapping/pool.json'),
        scores_hash=fingerprint(BASE/'overlapping/evaluation/padded/scores.json'),timeout_s=60,
        selected_records=selected,selected_hashes={p:fingerprint(p) for p in selected.values()})
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen protocol changed'
    write_json(path,protocol)
    rows=[]
    for i,(tree,key) in enumerate(zip(pool['trees'],pool['hashes'])):
        if key in selected:
            rec=read(selected[key]);row=dict(candidate_id=i,status=rec['status'],cx=rec.get('cx_count'),reused=True,record_path=selected[key])
        else:
            directory=OUT/f'candidate_{i}';directory.mkdir(exist_ok=True);done=directory/'controlled.json'
            if done.exists():rows.append(read(done));continue
            started=time.monotonic();process=mp.Process(target=worker,args=(target,canonical(tree),directory));process.start();process.join(60)
            timedout=process.is_alive()
            if timedout:process.terminate();process.join()
            record=directory/'record.json'
            rec=read(record) if record.exists() and not timedout else {}
            starts=directory/'attempt_starts.jsonl'
            row=dict(candidate_id=i,status='timeout' if timedout else rec.get('status','worker_error'),cx=rec.get('cx_count'),
                reused=False,record_path=str(record) if rec else None,wall_s=time.monotonic()-started,
                synthesis_starts=len(starts.read_text().splitlines()) if starts.exists() else 0,exitcode=process.exitcode)
            write_json(done,row)
        rows.append(row)
        write_json(OUT/'progress.json',dict(completed=len(rows),total=len(pool['trees'])))
        print(i,row['status'],row['cx'],'reused',row['reused'],flush=True)
    report(rows,read(BASE/'overlapping/evaluation/padded/scores.json'))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
