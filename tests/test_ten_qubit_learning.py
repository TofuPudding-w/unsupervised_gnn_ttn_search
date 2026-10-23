import numpy as np
import torch
from ttn_search.learning.encoder import Encoder,batch,loss
from ttn_search.trees import balanced,comb,features


def test_ten_qubit_dimensions_and_gradient():
    trees=[balanced(list(range(10))),comb(list(range(10)))]
    x,a=batch(trees);model=Encoder(n=10)
    assert x.shape==(2,19,15) and a.shape==(2,19,19)
    z=model(x,a);prediction=model.decoder(z)
    assert z.shape==(2,8) and prediction.shape==(2,55)
    truth=torch.tensor(np.array([features(t) for t in trees]),dtype=torch.float32)
    value=loss(prediction,truth,n=10);value.backward()
    assert model.input.weight.grad is not None
    assert torch.isfinite(model.input.weight.grad).all()


def test_ten_qubit_loss_balances_lca_and_leaf_groups():
    truth=torch.zeros((1,55));prediction=truth.clone()
    prediction[:,:45]=1
    assert loss(prediction,truth,n=10).item()==.5
    prediction[:,:45]=0;prediction[:,45:]=1
    assert loss(prediction,truth,n=10).item()==.5
