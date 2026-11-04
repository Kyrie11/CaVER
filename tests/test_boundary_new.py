import pytest
from caver.validation.boundary import (AnchorWorld,VerifiedPerturbation,assess_boundary,
                                       upper_transfer,potential_contrast,bounded_pair_loss)

def test_missing_guard_fails_closed():
    worlds=[AnchorWorld({'brake':3.},{'B|C':2.},'IDM')]*100
    env=VerifiedPerturbation({}, {'B|C':.1},True,True,'IDM')
    d=assess_boundary(worlds,env)
    assert not d.reuse and d.upper_exception_mass==1 and not d.valid

def test_complete_guard_and_contact_stable():
    worlds=[AnchorWorld({'brake':2.},{'B|C':3.},'IDM')]*1000
    env=VerifiedPerturbation({'brake':.1},{'B|C':.1},True,True,'IDM')
    d=assess_boundary(worlds,env,delta=.05,tolerance=.05)
    assert d.valid and d.reuse and d.empirical_exception_mass==0
    assert upper_transfer(.15,d) < .2

def test_near_guard_forces_recompute_and_unknown_policy():
    worlds=[AnchorWorld({'brake':.04},{'B|C':3.},'IDM')]*100
    env=VerifiedPerturbation({'brake':.1},{'B|C':.1},True,True,'IDM')
    assert assess_boundary(worlds,env).upper_exception_mass==1
    assert not assess_boundary(worlds,VerifiedPerturbation({'brake':.1},{'B|C':.1},True,True,'OTHER')).reuse

def test_potential_antisymmetry_and_slack_is_not_collision():
    assert potential_contrast(.7,.2)==pytest.approx(-potential_contrast(.2,.7))
    assert potential_contrast(.7,.2)+potential_contrast(.2,.3)==pytest.approx(potential_contrast(.7,.3))
    assert bounded_pair_loss(0,-10,True)==.25
    assert bounded_pair_loss(0,-10,False)==0
