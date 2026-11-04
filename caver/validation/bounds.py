"""Conservative cascade response tubes with explicit assumption checks.

A finite deterministic upper bound requires externally justified dynamical
response-gain and mode-switching envelopes. Neural Jacobians are *not* such
an upper bound. Statistical bounds are separately calibrated and labeled.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class ResponseEdge:
    sender:str
    receiver:str
    delay_steps:int
    gain:float
    independently_validated:bool

@dataclass
class PhysicalTube:
    node_displacement:dict
    valid:bool
    reason:str

def propagate_tube(ego_delta,actors,edges,local_gain=None,disturbance=None):
    """Each time step upper-bounds displacements via causal, delayed gains.

    Updates are simultaneous. A zero-delay feedback cycle is rejected because
    the simple recursion cannot justify a finite bound in that case.
    """
    e=np.asarray(ego_delta,float);T=len(e)
    if T==0 or not np.all(np.isfinite(e)) or np.any(e<0):raise ValueError('ego delta must be finite nonnegative')
    edges=list(edges);actors=list(actors)
    if 'E' not in actors:raise ValueError('Ego node E missing')
    if any(x.delay_steps<1 or x.gain<0 or not math.isfinite(x.gain) for x in edges):
        return PhysicalTube({},False,'gain/delay outside validated nonnegative envelope')
    if not all(x.independently_validated for x in edges):
        return PhysicalTube({},False,'unvalidated response gains (neural slopes are not physical upper bounds)')
    local_gain=local_gain or {a:1. for a in actors}
    disturbance=disturbance or {a:np.zeros(T) for a in actors}
    tracks={a:np.zeros(T) for a in actors};tracks['E']=e.copy()
    for t in range(1,T):
        for actor in actors:
            if actor=='E':continue
            own=local_gain.get(actor,float('nan'))
            if not math.isfinite(own) or own<0:return PhysicalTube({},False,'invalid local gain')
            val=own*tracks[actor][t-1]
            for edge in edges:
                if edge.receiver==actor and edge.sender in tracks and t-edge.delay_steps>=0:
                    val+=edge.gain*tracks[edge.sender][t-edge.delay_steps]
            noise=np.asarray(disturbance.get(actor,np.zeros(T)),float)[t]
            if not math.isfinite(noise) or noise<0:return PhysicalTube({},False,'invalid disturbance envelope')
            tracks[actor][t]=val+noise
            if not math.isfinite(tracks[actor][t]):return PhysicalTube({},False,'tube diverged')
    return PhysicalTube({k:v.tolist() for k,v in tracks.items()},True,'bounds conditional on validated gains, delays and modes')

def safety_boundary_error(tube,actor_a,actor_b,yaw_a=0.,yaw_b=0.,size_a=(4.5,1.8),size_b=(4.5,1.8)):
    if not tube.valid:return math.inf
    if actor_a not in tube.node_displacement or actor_b not in tube.node_displacement:return math.inf
    # max over time: translation error plus maximum footprint swing due to yaw.
    da=max(tube.node_displacement[actor_a]);db=max(tube.node_displacement[actor_b]);
    ra=.5*math.hypot(*size_a);rb=.5*math.hypot(*size_b)
    return da+db+2*ra*math.sin(min(abs(yaw_a),math.pi)/2)+2*rb*math.sin(min(abs(yaw_b),math.pi)/2)

def reuse_decision(anchor_margin,physical_error=None,statistical_error=None,require='statistical'):
    if require=='physical':
        if physical_error is None or not math.isfinite(physical_error):return 'UNKNOWN_PHYSICAL_BOUND'
        return 'REUSE_CERTIFIED_DECLARED_ENVELOPE' if anchor_margin>physical_error else 'RECOMPUTE'
    if require=='statistical':
        if statistical_error is None or not math.isfinite(statistical_error):return 'UNKNOWN_STATISTICAL_BOUND'
        return 'REUSE_REFERENCE_POPULATION' if anchor_margin>statistical_error else 'RECOMPUTE'
    raise ValueError('Unknown guarantee class')
