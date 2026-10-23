"""Read saved circuits; no synthesis or search reruns. Logical all-to-all depth."""
import json
import math
from pathlib import Path
from qiskit import qpy
from ttn_search.evaluate import write_json

OUT=Path('artifacts/stage_c1_depth')

def two_qubit_depth(circuit):
    """ASAP layers of two-qubit gates, with one-qubit durations ignored."""
    layers=[0]*circuit.num_qubits
    for instruction in circuit.data:
        n=instruction.operation.num_qubits
        if n>2:raise ValueError('Expected fully decomposed one-/two-qubit circuit')
        if n==2:
            a,b=[circuit.find_bit(q).index for q in instruction.qubits]
            layers[a]=layers[b]=max(layers[a],layers[b])+1
    return max(layers,default=0)

def read(p):return json.loads(Path(p).read_text())

def measure(record_path,cache):
    record=read(record_path);path=record['circuit_path']
    if path not in cache:
        with Path(path).open('rb') as f:circuit=qpy.load(f)[0]
        assert set(circuit.count_ops())<= {'u','cx'}
        depth=two_qubit_depth(circuit)
        assert depth==circuit.depth(filter_function=lambda x:x.operation.num_qubits==2)
        assert circuit.count_ops().get('cx',0)==record['cx_count']
        cache[path]=dict(cx=record['cx_count'],d2=depth,total_depth=circuit.depth(),fidelity=record['fidelity_final'],circuit_path=path)
    return cache[path]

def main():
    OUT.mkdir(parents=True,exist_ok=True);cache={}
    rows=[]
    for r in read('artifacts/stage_c1_headroom/results.json'):
        assert r['status']=='success'
        rows.append(dict(candidate_id=r['candidate_id'],**measure(r['record_path'],cache)))
    baseline=next(r for r in rows if r['candidate_id']==0)
    frontier=[r for r in rows if not any(a['cx']<=r['cx'] and a['d2']<=r['d2'] and (a['cx']<r['cx'] or a['d2']<r['d2']) for a in rows)]
    lower=[r for r in rows if r['d2']<baseline['d2']]
    caps=sorted(set([math.floor(.75*baseline['d2']),math.floor(.9*baseline['d2']),baseline['d2']-1,baseline['d2'],(11*baseline['d2']+9)//10]))
    lines=['# C1 logical two-qubit depth analysis','',
        '## Definition and scope','',
        'Read the accepted saved QPY circuits; no recompilation, routing, optimization or new search. Primary analysis covers the exhaustively evaluated 128-candidate overlapping pool at ten qubits, seed 13. All accepted circuits satisfy the original fidelity requirement 0.999.','',
        'D2 is the ASAP depth of the ordered two-qubit-gate subsequence: each CX is assigned one plus the maximum layer already used on its two wires. Disjoint CX gates can share a layer. One-qubit gates have zero duration for this metric. The result was independently checked against Qiskit filtered depth for every loaded circuit. Total logical depth including u gates is also recorded. This all-to-all structural metric is not hardware-scheduled execution time; no coupling-map routing or gate durations are included.','',
        '## Exhaustive pool results','',
        f"- SCENT: **{baseline['cx']} CNOTs, D2={baseline['d2']}, total logical depth={baseline['total_depth']}**.",
        f"- Pool D2 range: **{min(r['d2'] for r in rows)}–{max(r['d2'] for r in rows)}**.",
        f"- Candidates with smaller D2 than SCENT: **{len(lower)}**.",
        f"- Candidates with larger CNOT count but smaller D2: **{sum(r['cx']>baseline['cx'] for r in lower)}**.",
        f"- Nondominated candidates in (CNOT, D2): **{len(frontier)}**.",'',
        '| Frontier candidate | CNOT | D2 | Total logical depth | Fidelity |','|---|---:|---:|---:|---:|']
    for r in sorted(frontier,key=lambda r:r['cx']):lines.append(f"| {r['candidate_id']} | {r['cx']} | {r['d2']} | {r['total_depth']} | {r['fidelity']:.10f} |")
    lines+=['','## Exploratory depth caps','',
        'These caps are derived from SCENT depth (75%, 90%, one layer below, equal, and 110%) to describe the existing pool. They are development analyses, not prespecified independent tests. Feasibility here requires both the original fidelity constraint and D2 ≤ cap.','',
        '| D2 cap | Feasible candidates | Lowest feasible CNOT | SCENT eligible |','|---|---:|---:|---|']
    for cap in caps:
        eligible=[r for r in rows if r['d2']<=cap]
        lines.append(f"| {cap} | {len(eligible)} | {min((r['cx'] for r in eligible),default='none')} | {baseline['d2']<=cap} |")
    lines+=['','## Retrospective method comparison','',
        'Original overlapping search selections are held fixed; only their incumbent feasibility is recomputed. This is not a depth-aware search rerun. Fixed-evaluation and 30-second runs are separate experiments. Deadline-interrupted work remains excluded. No ten-qubit GNN was evaluated.','',
        '| Mode | Method | Cap | Best at K=1 | K=4 | K=16 | Best at run endpoint |','|---|---|---:|---:|---:|---:|']
    replay=[]
    for run in read('artifacts/stage_c1/runs.json'):
        if run['family']!='overlapping':continue
        measured=[(t,measure(t['record_path'],cache)) for t in run['trace'] if t['status']=='success']
        for cap in [baseline['d2']-1,baseline['d2']]:
            def best(k):return min((m['cx'] for t,m in measured if t['k']<=k and m['d2']<=cap),default=None)
            values=[best(k) for k in [1,4,16,len(run['trace'])]]
            lines.append(f"| {run['mode']} | {run['method']} | {cap} | "+' | '.join(str(v) if v is not None else 'none' for v in values)+' |')
            replay.append(dict(mode=run['mode'],method=run['method'],cap=cap,values=values))
    if not lower:
        conclusion='SCENT minimizes both CNOT count and two-qubit depth in this fixed pool. A depth cap below its D2 leaves no feasible candidate; a cap at or above it retains SCENT as the lowest-CNOT choice. Therefore adding a depth constraint does not create improvement headroom here. This conclusion does not extend to unseen pools, target instances, other compilers, or routed hardware circuits.'
    else:
        conclusion='The pool contains genuine count–depth tradeoffs. A depth constraint can change which candidates are feasible. Freeze development-derived caps before a new target benchmark, and compare depth-aware selection with the same baselines and evaluation accounting.'
    lines+=['','## Interpretation and next experiment','',conclusion,'',
        'Yes, compare methods under a depth constraint in a broader benchmark, alongside the original unconstrained resource comparison. Keep CNOT count and D2 separate, report infeasibility explicitly, and choose caps from development data or hardware requirements before evaluating untouched targets. A depth-aware BO must model or otherwise incorporate the depth constraint; filtering its previous trace does not establish its prospective constrained performance.','']
    write_json(OUT/'pool_depths.json',rows);write_json(OUT/'replay.json',replay)
    write_json(OUT/'audit.json',dict(candidates=len(rows),loaded_circuits=len(cache),qiskit_depth_crosscheck=True,baseline=baseline,pareto_ids=[r['candidate_id'] for r in frontier],lower_depth_count=len(lower)))
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter([r['cx'] for r in rows],[r['d2'] for r in rows],alpha=.6,label='Pool candidates')
    ax.scatter([baseline['cx']],[baseline['d2']],marker='*',s=180,color='red',label='SCENT')
    ax.set(xlabel='CNOT count',ylabel='Logical two-qubit depth',title='10-qubit overlapping pool: 128 candidates')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'count_depth.png',dpi=180);plt.close(fig)
    print(conclusion)

if __name__=='__main__':main()
