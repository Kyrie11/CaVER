import numpy as np
import torch
from caver.models.cbc import CascadeRiskHead,FEATURES,training_loss,all_downstream_pairs,pair_features
from caver.models.policy import LoggedPolicyHead
from caver.sim.traffic import standard_fourcar,candidates,sample_world,rollout

def test_cbc_forward_and_gradient():
    m=CascadeRiskHead(width=24)
    X=torch.zeros(4,len(FEATURES));X[:,9]=.3
    z=m(X)
    assert z.shape==(4,3)
    assert torch.all((z[:,0]>0)&(z[:,0]<1))
    y=torch.tensor([[.6,2.,5.,1.]]*4)
    l=training_loss(z,y);l.backward()
    assert m.output.weight.grad is not None

def test_cbc_source_features_reproducible():
    s=standard_fourcar();a=candidates()[0]
    samples=[rollout(s,a,sample_world(s,12))]
    x=pair_features(s,a,samples,'B|C')
    assert x.shape==(len(FEATURES),)
    assert np.all(np.isfinite(x))

def test_policy_network_is_separate_from_smart():
    m=LoggedPolicyHead()
    assert m(torch.zeros(3,5)).shape==(3,)
