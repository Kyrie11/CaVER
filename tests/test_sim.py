import numpy as np
from caver.sim.traffic import standard_fourcar,candidates,sample_world,rollout
from caver.experiment import mediator_exchange

def test_joint_rollout_physics_and_roles():
    s=standard_fourcar()
    w=sample_world(s,8,'delayed_idm')
    r=rollout(s,candidates()[0],w,horizon=1.5)
    assert r['trajectory'].shape==(16,4,6)
    assert np.all(r['trajectory'][:,:,2]>=0)
    assert r['is_surrogate']
    assert r['trajectory'][-1,0,1]>r['trajectory'][0,0,1]
    assert 'B|C' in r['contact']

def test_mediator_swap_is_not_field_identification():
    s=standard_fourcar()
    out=mediator_exchange(s,sample_world(s,4,'late_brake'))
    assert out['physical_consistency_replay_not_guaranteed']
    assert 'a_with_b_mediator' in out

def test_shared_world_yields_identical_rollout():
    s=standard_fourcar();a=candidates()[1];w=sample_world(s,20)
    x=rollout(s,a,w,horizon=1)
    y=rollout(s,a,w,horizon=1)
    np.testing.assert_array_equal(x['trajectory'],y['trajectory'])
