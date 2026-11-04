import math
import numpy as np
import pytest
from caver.sim.geometry import obb_signed_clearance,trajectory_clearance,stop_slack,crossing_bounds

def test_rect_overlap():
    a=(0.,0.,0.,4.,2.);b=(2.,0.,0.,4.,2.)
    assert obb_signed_clearance(a,b)<0
    assert obb_signed_clearance(a,(10.,0.,0.,4.,2.))>0
    assert obb_signed_clearance(a,(0.,4.,0.,4.,2.))>0

def test_frechet_sharp_bounds():
    b=crossing_bounds(.8,.1)
    assert b['harm_lower']==pytest.approx(.7)
    assert b['flip_lower']==pytest.approx(.7)
    assert b['flip_upper']==pytest.approx(.9)
    with pytest.raises(ValueError):crossing_bounds(1.2,.2)

def test_stop_slack_physical_limits():
    assert stop_slack(50,13,13,.2,0,6,6)>0
    assert stop_slack(2,19,10,.8,1,2,4)<0
    assert stop_slack(20,10,10,.4,0,0,6)==-math.inf

def test_swept_require_motion_bound():
    a=np.array([[0,0,0,4,2],[10,0,0,4,2]],float)
    b=np.array([[5,0,0,4,2],[5,0,0,4,2]],float)
    with pytest.raises(ValueError):trajectory_clearance(a,b,.1,strict=True)


def test_vectorized_obb_clearance_matches_scalar_many_yaws():
    rng=np.random.default_rng(2148)
    A=np.column_stack([rng.normal(size=128)*8, rng.normal(size=128)*3,
                       rng.uniform(-math.pi,math.pi,128),rng.uniform(3.,6.,128),rng.uniform(1.,3.,128)])
    B=np.column_stack([rng.normal(size=128)*8, rng.normal(size=128)*3,
                       rng.uniform(-math.pi,math.pi,128),rng.uniform(3.,6.,128),rng.uniform(1.,3.,128)])
    expected=min(obb_signed_clearance(a,b) for a,b in zip(A,B))
    assert trajectory_clearance(A,B)==pytest.approx(expected,abs=1e-11)
    assert trajectory_clearance(A,B,max_speed=10,dt=.1)==pytest.approx(expected-.5,abs=1e-11)
    with pytest.raises(ValueError):trajectory_clearance(A[:1],B)
