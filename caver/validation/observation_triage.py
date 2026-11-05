"""Observation-only CBC prescreen: consumes <=t0 WOMD history; never ground-truth future.

The output is a compact population-level quality report, not a counterfactual
estimate, event rate, model accuracy, or proof of legal lane-change feasibility.
"""
import math
from collections import Counter
from pathlib import Path
import numpy as np
from caver.data.schema import read_jsonl
from caver.sim.geometry import obb_signed_clearance,stop_slack


def _valid_accelerations(a,dt=.1):
    if not a.history:return []
    rows=a.history
    out=[]
    for i in range(1,len(rows)):
        if rows[i][5] and rows[i-1][5]:
            v0=math.hypot(rows[i-1][2],rows[i-1][3]);v1=math.hypot(rows[i][2],rows[i][3])
            out.append((i, (v1-v0)/dt))
    return out


def _reasons(sc):
    if not sc.provenance.get('history_only',False) or sc.provenance.get('future_trajectory_exposed',True):
        raise ValueError('Observational CBC triage refuses data containing future trajectory')
    agents={a.agent_id:a for a in sc.agents}
    if not {'A','B','C','E'} <= set(agents):
        raise ValueError('Triage requires pre-normalized four-car corridor scenes')
    E,B,C=agents['E'],agents['B'],agents['C']
    # All comparisons made at the planning instant. No counterfactual or
    # future data, not even the logged training/validation future, is loaded.
    lead,rear=(B,C) if B.x>=C.x else (C,B)
    gap=lead.x-rear.x-.5*(lead.length+rear.length)
    relspeed=math.hypot(rear.vx,rear.vy)-math.hypot(lead.vx,lead.vy)
    speed=math.hypot(rear.vx,rear.vy)
    thw=gap/max(speed,.1)
    ttc=gap/relspeed if relspeed>.1 and gap>0 else None
    physical={}
    for tau in (.4,.8,1.2):
        physical[str(tau)]=stop_slack(gap,speed,math.hypot(lead.vx,lead.vy),tau,
                     1.2,3.5,3.5)
    b_ac=_valid_accelerations(B,sc.dt)
    c_ac=_valid_accelerations(C,sc.dt)
    def brake(xs,threshold=-2.):return any(x<threshold for _,x in xs)
    e_b=obb_signed_clearance((E.x,E.y,E.yaw,E.length,E.width),
                             (B.x,B.y,B.yaw,B.length,B.width))
    b_c=obb_signed_clearance((B.x,B.y,B.yaw,B.length,B.width),
                             (C.x,C.y,C.yaw,C.length,C.width))
    return {'bc_gap_m':gap,'bc_thw_s':thw,'bc_closing_mps':relspeed,
       'bc_ttc_s':ttc,'bc_bounded_slack_proxy_m':physical,
       'bc_initial_obb_signed_clearance_m':b_c,
       'eb_initial_obb_signed_clearance_m':e_b,
       'ego_velocity_mps':math.hypot(E.vx,E.vy),
       'b_braking_in_observed_history':brake(b_ac),
       'c_braking_in_observed_history':brake(c_ac),
       'b_history_acceleration_samples':len(b_ac),
       'c_history_acceleration_samples':len(c_ac),
       'map_preserved':bool(sc.map_data and sc.map_data.get('polylines')),
       'overlap_group_quality':sc.provenance.get('overlap_group_quality','UNKNOWN'),
       'source_split':sc.provenance.get('split','UNKNOWN'),
       'initial_bc_contact':b_c<=0,'initial_ego_b_contact':e_b<=0}


def _compact(v):
    v=[float(x) for x in v if x is not None and np.isfinite(x)]
    if not v:return {'n':0,'median':None,'p05':None,'p25':None,'p75':None,'p95':None}
    return {'n':len(v),'median':float(np.median(v)),'p05':float(np.quantile(v,.05)),
            'p25':float(np.quantile(v,.25)),'p75':float(np.quantile(v,.75)),
            'p95':float(np.quantile(v,.95))}


def observation_audit(input_path,limit=None):
    """One summary for a frozen natural or geometry-filtered population.

    No selection is based on future labels or reference reactor outcomes.
    """
    if limit is not None and limit<1:raise ValueError('limit must be positive')
    metrics=[];splits=Counter();groups=set();errors=Counter()
    for sc in read_jsonl(input_path,limit):
        try:result=_reasons(sc)
        except ValueError as err:
            errors[str(err)]+=1
            continue
        metrics.append(result)
        splits[result['source_split']]+=1
        groups.add(sc.provenance.get('overlap_group',sc.scene_id))
    fields=['bc_gap_m','bc_thw_s','bc_closing_mps','bc_ttc_s',
            'bc_initial_obb_signed_clearance_m','eb_initial_obb_signed_clearance_m',
            'ego_velocity_mps','b_history_acceleration_samples','c_history_acceleration_samples']
    numerical={f:_compact([r[f] for r in metrics]) for f in fields}
    numerical['stop_slack_delay_0p4_m']=_compact([r['bc_bounded_slack_proxy_m']['0.4'] for r in metrics])
    numerical['stop_slack_delay_0p8_m']=_compact([r['bc_bounded_slack_proxy_m']['0.8'] for r in metrics])
    numerical['stop_slack_delay_1p2_m']=_compact([r['bc_bounded_slack_proxy_m']['1.2'] for r in metrics])
    n=len(metrics)
    def fr(fn):return float(sum(bool(fn(x)) for x in metrics)/n) if n else None
    return {'schema':'CBC_OBSERVATION_ONLY_CONTEXT_V1','input':str(input_path),
      'observed_scenes':n,'unique_provisional_overlap_groups':len(groups),
      'splits':dict(splits),'decode_rejected':dict(errors),
      'quantiles':numerical,
      'fractions':{
        'initial_BC_contact':fr(lambda r:r['initial_bc_contact']),
        'initial_Ego_B_contact':fr(lambda r:r['initial_ego_b_contact']),
        'BC_time_headway_under_1s':fr(lambda r:r['bc_thw_s']<1),
        'BC_positive_closure':fr(lambda r:r['bc_closing_mps']>0),
        'BC_TTC_under_3s_of_positive_closure':fr(lambda r:r['bc_ttc_s'] is not None and 0<r['bc_ttc_s']<3),
        'BC_stop_slack_negative_at_0p8s':fr(lambda r:r['bc_bounded_slack_proxy_m']['0.8']<0),
        'B_observed_braking_under_minus_2mps2':fr(lambda r:r['b_braking_in_observed_history']),
        'C_observed_braking_under_minus_2mps2':fr(lambda r:r['c_braking_in_observed_history']),
        'map_polyline_available_after_corridor_filter':fr(lambda r:r['map_preserved'])},
      'population':'GEOMETRICALLY_PRESELECTED_WOMD_CORRIDORS_NOT_NATURAL_FREQUENCY',
      'caveats':['All inputs are observation history and current state, not WOMD logged future',
          'Rear-end slack assumes hypothetical fixed deceleration and delay, not measured driver capability',
          'Collision/secondary risk requires two independent action-conditioned reactive rollouts',
          'If map is not retained, lawful lane-change matching is not established',
          'Short historical windows can miss prior braking events and response onsets']}
