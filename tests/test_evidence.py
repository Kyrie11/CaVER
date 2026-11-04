from caver.validation.evidence import conformal_calibrate,choose
from caver.validation.bounds import propagate_tube,ResponseEdge,safety_boundary_error,reuse_decision
from caver.validation.tiv import propose

def _pred(v=.01):
    return {f'{a}/{field}':v for a in ('a','b','reference') for field in ('ego','down','B|C')}

def test_zero_radius_admission_and_regret():
    from caver.validation.evidence import Calibration
    cal=Calibration(0.,.1,100,1000,.1,0.,['ref'],'ABC','TEST')
    d=choose(_pred(),['a','b','reference'],['B|C'],calibration=cal,
             eps_ego=.1,eps_down=.1,eps_inc=.1,eps_reg=1,strict_prefix=True)
    assert d.status=='DECISION-CERTIFIED'
    assert d.released and d.gamma_plausible['a'] >= d.gamma_robust['a']

def test_uncalibrated_cannot_certify():
    d=choose(_pred(),['a','b','reference'],['B|C'])
    assert d.status=='UNCERTIFIED'
    assert d.chosen is None

def test_reference_relative_increment_not_absolute_risk():
    from caver.validation.evidence import Calibration
    cal=Calibration(0.,.1,100,1000,.1,0.,['ref'],'ABC','TEST')
    v=_pred(.4);v['a/B|C']=.41
    d=choose(v,['a','b','reference'],['B|C'],calibration=cal,eps_ego=.5,eps_down=.05,strict_prefix=True)
    assert not d.candidates

def test_whole_scene_conformal_is_explicit_about_small_n():
    x=[{'predicted':_pred(.1),'audited':_pred(.2)} for _ in range(6)]
    cal=conformal_calibrate(x,alpha=.1,audit_worlds=16)
    assert cal.quantile==1.
    assert cal.mc_buffer>0

def test_tube_chain_and_invalid_gain():
    e=[0,1,1,1,1,1]
    edges=[ResponseEdge('E','B',1,2,True),ResponseEdge('B','C',1,3,True)]
    x=propagate_tube(e,['E','B','C'],edges,{'B':.8,'C':.7})
    assert x.valid
    assert x.node_displacement['C'][-1]>0
    assert safety_boundary_error(x,'B','C')>=0
    y=propagate_tube(e,['E','B','C'],[ResponseEdge('B','C',1,3,False)])
    assert not y.valid
    assert reuse_decision(5,physical_error=None,require='physical')=='UNKNOWN_PHYSICAL_BOUND'

def test_query_targets_critical_pairs():
    d=choose(_pred(),['a','b','reference'],['B|C'])
    q=propose(d,_pred(),['a','b','reference'],['B|C'])
    assert q and q[0].pairs==('B|C',)
