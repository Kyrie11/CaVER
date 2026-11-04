"""Falsification tests: no future observations, no false DSE decisions."""
from types import ModuleType, SimpleNamespace as NS
import sys
import numpy as np
import pytest
import torch
from caver.data.schema import Agent
from caver.data.corridor import corridor_subset
from caver.data.womd import parse_tf_example
from caver.sim.traffic import standard_fourcar,sample_world,rollout,candidates
from caver.validation.evidence import Calibration,choose
from caver.validation.boundary import frechet_flip_bounds,frechet_harm_bounds
from caver.validation.tiv import propose,select_query,Query


def fixture_cal(q=0.):
    return Calibration(q,.1,1000,1000,.05,0.,['REFERENCE'],'D','TRAIN_CAL_ONLY')


def test_plausible_regret_detects_risk_only_vs_certified():
    pred={}
    for a,base in (('a',.01),('b',.085),('reference',.01)):
        for k in ('ego','down','B|C'):
            pred[f'{a}/{k}']=base if k!='ego' else .01
    common=dict(calibration=fixture_cal(.01),eps_ego=.1,eps_down=.085,
                eps_inc=.4,eps_reg=1.,strict_prefix=True,
                task_cost={'a':5.,'b':0.,'reference':10.},lambda_contact=0.)
    d=choose(pred,['a','b','reference'],['B|C'],**common)
    assert d.status=='RISK-ADMISSIBLE'
    assert d.chosen is None and not d.released
    assert 'a' in d.candidates and 'b' not in d.candidates
    assert d.gamma_robust['a'] == pytest.approx(0.)
    assert d.gamma_plausible['a'] >= 5.
    assert d.gamma_plausible['a'] > d.gamma_robust['a']
    d2=choose(pred,['a','b','reference'],['B|C'],allow_risk_only=True,**common)
    assert d2.status=='RISK-ADMISSIBLE' and d2.released and d2.chosen=='a'


def test_prefix_and_calibration_protocol_fail_closed():
    pred={f'{a}/{k}':.01 for a in ('a','reference') for k in ('ego','down','B|C')}
    args=dict(actions=['a','reference'],pairs=['B|C'],calibration=fixture_cal(),eps_reg=1.)
    d=choose(pred,**args,strict_prefix=False)
    assert d.status=='UNCERTIFIED' and not d.released
    d=choose(pred,**args,strict_prefix=True,procedure='AFTER_AN_UNCALIBRATED_QUERY')
    assert d.status=='UNCERTIFIED' and not d.released
    p=dict(pred);p['a/down']=.05;p['a/B|C']=.8
    d=choose(p,**args,strict_prefix=True)
    assert d.status=='UNCERTIFIED' and 'empty coherent' in d.reason


def test_no_unverified_slack_constraint_can_pass():
    p={f'{a}/{k}':.01 for a in ('a','reference') for k in ('ego','down','B|C')}
    d=choose(p,['a','reference'],['B|C'],calibration=fixture_cal(),
             eps_ego=.1,eps_down=.1,eps_inc=.1,eps_slack=.1,strict_prefix=True)
    assert not d.candidates and d.status=='UNCERTIFIED'


def test_frechet_marginal_bounds_without_paired_cross_world_leakage():
    assert frechet_flip_bounds(.8,.2)==pytest.approx((.6,1.))
    assert frechet_harm_bounds(.8,.2)==pytest.approx((.6,.8))
    assert frechet_harm_bounds(.2,.8)==pytest.approx((0.,.2))


def test_corridor_preserves_valid_histories_in_local_coordinates():
    s=standard_fourcar()
    next(a for a in s.agents if a.agent_id=='E').x=8.
    for a in s.agents:
        a.history=[[a.x-1.,a.y,a.vx,a.vy,a.yaw,1],
                   [a.x,a.y,a.vx,a.vy,a.yaw,1]]
    c,reason=corridor_subset(s)
    assert c is not None,reason
    assert all(len(a.history)==2 for a in c.agents)
    for a in c.agents:
        assert a.history[-1][0] == pytest.approx(a.x)
        assert a.history[-1][1] == pytest.approx(a.y)
        assert a.history[0][0] == pytest.approx(a.x-1.)


def test_delayed_reactor_uses_observed_past_not_repeated_current():
    from copy import deepcopy
    a=standard_fourcar();b=deepcopy(a)
    for sc in (a,b):
        for ag in sc.agents:
            ag.history=[[ag.x,ag.y,ag.vx,ag.vy,ag.yaw,1] for _ in range(11)]
    # Only change past A/B positions; snapshot at t0 is identical.
    for ag in b.agents:
        if ag.agent_id=='B':
            for h in ag.history[:-1]:h[0] += 25.
    assert [x.x for x in a.agents]==[x.x for x in b.agents]
    wa=sample_world(a,12);wb=sample_world(b,12)
    ra=rollout(a,candidates()[0],wa,horizon=.5)
    rb=rollout(b,candidates()[0],wb,horizon=.5)
    assert abs(ra['accelerations']['C'][0]-rb['accelerations']['C'][0])>1e-5


def test_tiv_requires_held_prefix_and_positive_signed_value():
    pred={f'{a}/{k}':.01 for a in ('a','reference') for k in ('ego','down','B|C')}
    d=choose(pred,['a','reference'],['B|C'])
    q=Query(('a',),('B|C',),('ego','B','C'),(0.,4.),'more_source_samples',2,2.,10.)
    class Positive(torch.nn.Module):
        def forward(self,x):return torch.ones(len(x))
    assert select_query(d,pred,[q],Positive(),holding_prefix_verified=False)['query'] is None
    assert select_query(d,pred,[q],Positive(),holding_prefix_verified=True,deadline_ms=5.)['query'] is None
    assert select_query(d,pred,[q],Positive(),holding_prefix_verified=True)['query']==q
    assert select_query(d,pred,[q],holding_prefix_verified=True)['query'] is None


def test_tf_example_uses_float_ids_and_no_future_by_default(monkeypatch):
    class FakeF:
        def __init__(self,kind,vals):
            self.kind=kind;setattr(self,kind,NS(value=vals))
        def HasField(self,x):return x==self.kind
    values={}
    def put(k,t,v):values[k]=FakeF(t,v)
    for k,vals in [('x',[0.,5.]),('y',[0.,3.5]),('velocity_x',[10.,11.]),
                   ('velocity_y',[0.,0.]),('bbox_yaw',[0.,0.]),('length',[4.5,4.4]),
                   ('width',[2.,2.])]:put('state/current/'+k,'float_list',vals)
    put('state/current/valid','int64_list',[1,1])
    put('state/type','float_list',[1.,1.])
    put('state/id','float_list',[123.,456.])
    put('state/is_sdc','int64_list',[1,0])
    put('scenario/id','bytes_list',[b'fake-scene'])
    for k,vals in [('x',[-2.,-1.,3.,4.]),('y',[0.,0.,3.5,3.5]),
                   ('velocity_x',[10.,10.,11.,11.]),('velocity_y',[0.,0.,0.,0.]),
                   ('bbox_yaw',[0.,0.,0.,0.])]:put('state/past/'+k,'float_list',vals)
    put('state/past/valid','int64_list',[1,1,1,1])
    # This is a deliberately forbidden future payload.
    put('state/future/x','float_list',[9999.,9999.])
    put('state/future/y','float_list',[8888.,8888.])
    put('state/future/valid','int64_list',[1,1])
    fake_train=ModuleType('tensorflow.train')
    class Example:
        @staticmethod
        def FromString(raw):return NS(features=NS(feature=values))
    fake_train.Example=Example
    fake_tf=ModuleType('tensorflow');fake_tf.train=fake_train
    monkeypatch.setitem(sys.modules,'tensorflow',fake_tf)
    monkeypatch.setitem(sys.modules,'tensorflow.train',fake_train)
    s,future=parse_tf_example(b'fake',include_future=False)
    assert s.agents[0].agent_id=='123' and s.agents[0].history[0][0]==-2
    assert future=={}
    assert '9999.0' not in str(s.to_json())
    assert s.provenance['future_trajectory_exposed'] is False
    s,labels=parse_tf_example(b'fake',include_future=True)
    assert labels['123'][0][0]==9999.
    assert all('9999.0' not in str(a.history) for a in s.agents)
