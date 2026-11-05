"""Frozen, t<=0 candidate generation for CBC mechanism studies.

No prediction labels or simulated reactions are used for candidate selection.
The screen uses an explicitly weak constant-velocity, constant-heading
extrapolation for surrounding agents; it is NOT lawful-road or causal safety.
Maintains fixed `early`, `late`, `reference` names for matched-pair metrics.
"""
from dataclasses import asdict
from itertools import product
import numpy as np
from caver.sim.traffic import Action,prescribed_ego_path
from caver.sim.geometry import trajectory_clearance


def _projected_non_ego(sc,agent,horizon=8.,dt=.1):
    t=np.arange(int(round(horizon/dt))+1)*dt
    # Corridor proxy assumes straight constant velocity. For curves or turns
    # this extrapolation must be replaced by a topology-aware prior.
    pos=np.stack((agent.x+agent.vx*t,agent.y+agent.vy*t),axis=1)
    return np.column_stack((pos,np.full(len(t),agent.yaw),
                          np.full(len(t),agent.length),np.full(len(t),agent.width)))


def _ego_geo(tr):
    return tr[:,[0,1,3,4,5]]


def choose_observation_only_actions(sc):
    """Select two near-intent ego controls without accessing other-agent futures.

    Returns a deterministic first-stage design, never a guarantee of safety.
    Candidate list and all policy hyperparameters are frozen before evaluation.
    We intentionally preserve unselectable scenes in the report; no hindsight
    filtering on downstream collision is permitted.
    """
    actors={a.agent_id:a for a in sc.agents}
    if sc.ego_id not in actors or 'B' not in actors:
        raise ValueError('Adaptive candidate policy requires corridor E/B roles')
    E=actors[sc.ego_id];B=actors['B']
    other=[a for a in sc.agents if a.agent_id!=sc.ego_id]
    frozen_other={a.agent_id:_projected_non_ego(sc,a) for a in other}
    best=None
    trials=0
    # Candidate variation changes longitudinal braking timing but not the
    # lane/gap target. Pulses have smooth ramps to satisfy jerk proxies.
    for lane_time,decel,pulse_start,sep in product((3.0,4.0),(.7,1.1,1.5),
                                                   (.4,.8,1.2,1.6),(.6,.9)):
        a=Action('early',pulse_start,pulse_start+1.2,decel,lane_offset=3.5,
                 lane_time=lane_time,accel_ramp_s=.6)
        b=Action('late',pulse_start+sep,pulse_start+sep+1.2,decel,
                 lane_offset=3.5,lane_time=lane_time,accel_ramp_s=.6)
        ta=prescribed_ego_path(sc,a);tb=prescribed_ego_path(sc,b)
        # Exact prescribed ego trajectories only; not simulated other agents.
        d=float(np.mean(np.linalg.norm(ta[:,:2]-tb[:,:2],axis=1)/5.
            +np.abs(ta[:,2]-tb[:,2])/4.
            +np.r_[np.abs(np.diff(ta[:,2]-tb[:,2])/sc.dt)/3.,0.]))
        if d>1.8:continue
        trial_clear=[]
        for path in (ta,tb):
            geo=_ego_geo(path)
            trial_clear.append(min(trajectory_clearance(geo,other_path)
                    for other_path in frozen_other.values()))
        # Both actions nominally safe under constant-velocity screening is
        # preferable, but does not eliminate all failed scenes.
        safe_min=min(trial_clear)
        # Relative x change can affect the gap at which B first encounters ego.
        t=np.arange(len(ta))*sc.dt
        mid=(t>=.7)&(t<=4.0)
        b_future=B.x+B.vx*t
        proximity=np.exp(-np.abs(ta[:,0]-b_future)/12.)
        prox_b=np.exp(-np.abs(tb[:,0]-b_future)/12.)
        influence=float(np.mean(np.abs(proximity[mid]-prox_b[mid])))
        # Large predicted contact is penalized without using a response model;
        # near guard differences are rewarded. No action risk truth is used.
        score=(4.0*float(safe_min>.5)+min(safe_min,1.5)/10.
               +.3*influence-.025*abs(ta[-1,0]-tb[-1,0]))
        key=(score,-trials)
        if best is None or key>best[0]:best=(key,(a,b),safe_min,d,influence)
        trials+=1
    if best is None:
        raise ValueError('Could not construct any matched smooth ego controls from current observation')
    _,ab,min_clear,d,influence=best
    a0=Action('reference',9.,10.,0.,lane_offset=0.,lane_time=3.5,accel_ramp_s=.6)
    return [*ab,a0],{'policy':'OBSERVATION_ONLY_ADAPTIVE_V1',
       'source':'CURRENT_STATE_ONLY_CONSTANT_VELOCITY_SCREEN_NOT_LAWFUL_OR_SAFE',
       'tested_predeclared_templates':trials,
       'predicted_min_ego_non_ego_clearance_constant_velocity_m':float(min_clear),
       'ego_candidate_distance_proxy':float(d),
       'B_candidate_interaction_contrast_proxy':float(influence),
       'map_law_route_feasibility':'UNVERIFIED',
       'controls':[asdict(a) for a in ab]}
