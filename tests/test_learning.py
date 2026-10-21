import pytest
torch = pytest.importorskip("torch")
from ttn_search.learning.encoder import Encoder, batch, corpus, loss
from ttn_search.trees import comb, tree_hash


def test_graph_invariance_and_labels():
    torch.manual_seed(0)
    model = Encoder().eval()
    tree = comb(list(range(8)))
    assert isinstance(tree, tuple)
    x, a = batch([tree])
    p = torch.randperm(15)
    with torch.no_grad():
        z = model(x,a)
        torch.testing.assert_close(z,model(x[:,p],a[:,p][:,:,p]),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(z,model(*batch([(tree[1],tree[0])])),atol=1e-6,rtol=1e-5)
        assert not torch.allclose(z,model(*batch([comb(list(reversed(range(8))))])),atol=1e-7)
        assert torch.isfinite(z).all()


def test_corpus_unique_excluded_and_reproducible():
    forbidden = {tree_hash(t) for t in corpus(5,set())}
    a = corpus(50,forbidden)
    assert a == corpus(50,forbidden)
    hashes = {tree_hash(t) for t in a}
    assert len(hashes)==50 and not hashes & forbidden


def test_reconstruction_loss_weights_groups_equally():
    truth = torch.zeros(2,36)
    p = truth.clone(); p[:,:28] = 1
    assert float(loss(p,truth)) == .5
    p = truth.clone(); p[:,28:] = 1
    assert float(loss(p,truth)) == .5


def test_replay_only_supplies_observed_labels(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('stage_b_runner', Path('scripts/run_stage_b.py'))
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    pool = corpus(12,set())
    records = [dict(status='success', fidelity_final=1., cx_count=100-i,
                    evaluation_time_s=1., evaluation_key=str(i)) for i in range(12)]
    calls = []
    def spy(x, observed, unseen, threshold, rng):
        assert not set(observed) & set(unseen)
        assert len(observed)+len(unseen)==12
        calls.append(len(observed))
        return unseen[0], {'acquisition':'test'}
    monkeypatch.setattr(runner,'propose',spy)
    trace = runner.replay(pool,records,None,'gnn',0,budget=10)
    assert calls == [8,9]
    assert len({r['candidate_id'] for r in trace})==10
    assert trace[-1]['charged_evaluation_time_s']==10
    seen = [r['candidate_id'] for r in trace]
    assert trace[-1]['best_cx']==min(records[i]['cx_count'] for i in seen)
