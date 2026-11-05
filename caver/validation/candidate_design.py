"""Candidate-design triage with NO reactive prediction or future labels.

Use only observation-conditioned ego trajectories and simple constant-velocity
agent forecasts. A selected pair is a *candidate hypothesis*, not lawful or
safe. This allows fixing sample selection before costly reference rollouts.
"""
from collections import Counter
import numpy as np
from caver.data.schema import read_jsonl
from caver.sim.traffic import candidates,prescribed_ego_path
from caver.sim.geometry import trajectory_clearance
from caver.models.candidate_policy import choose_observation_only_actions,_projected_non_ego


def _assess(sc,actions):
    a,b=actions[0],actions[1]
    assert (a.action_id,b.action_id)==('early','late')
    ta=prescribed_ego_path(sc,a);tb=prescribed_ego_path(sc,b)
    dt=sc.dt
    da=np.diff(ta[:,2])/dt;db=np.diff(tb[:,2])/dt
    ja=np.diff(da)/dt;jb=np.diff(db)/dt
    la=np.diff(ta[:,1],n=2)/dt**2;lb=np.diff(tb[:,1],n=2)/dt**2
    dyn=(max(abs(da))<=3.0+1e-4 and max(abs(db))<=3.0+1e-4
         and max(abs(ja))<=5.0+1e-4 and max(abs(jb))<=5.0+1e-4
         and max(abs(la))<=3.0+1e-4 and max(abs(lb))<=3.0+1e-4)
    d=float(np.mean(np.linalg.norm(ta[:,:2]-tb[:,:2],axis=1)/5.
       +abs(ta[:,2]-tb[:,2])/4.
       +np.r_[abs(da-db)/3.,0.]))
    J=lambda tr,da:-(tr[-1,0]-tr[0,0])/10. + .04*np.sum(da**2)*dt
    task=abs(J(ta,da)-J(tb,db))
    intent=(abs(a.lane_offset-b.lane_offset)<.1 and abs(a.lane_time-b.lane_time)<.1)
    reasons=[]
    if not intent:reasons.append('MISMATCH_INTENT')
    if not dyn:reasons.append('DYNAMICS')
    if d>1.8:reasons.append('TRAJECTORY_DISTANCE')
    if task>3.0:reasons.append('TASK_COST_DIFFERENCE')
    ext=[_projected_non_ego(sc,x) for x in sc.agents if x.agent_id!=sc.ego_id]
    clear=[]
    for tr in (ta,tb):
        geo=tr[:,[0,1,3,4,5]]
        clear.append(min(trajectory_clearance(geo,other) for other in ext))
    return {'matched_kinematic_proxy':not reasons,'reasons':reasons,
            'd_e':d,'task_difference':task,'min_nominal_clearance_m':float(min(clear)),
            'both_clear_constant_velocity':bool(min(clear)>.5),
            'map_law_verified':False}


def study(input_path,limit=None):
    scenes=list(read_jsonl(input_path,limit=limit))
    methods={'fixed':[],'observed_adaptive':[]}
    rejected=Counter()
    for sc in scenes:
        if not sc.provenance.get('history_only') or sc.provenance.get('future_trajectory_exposed',True):
            raise ValueError('Candidate-design triage requires past/current-only input')
        methods['fixed'].append(_assess(sc,candidates()))
        try:actions,_=choose_observation_only_actions(sc)
        except ValueError as exc:
            rejected[str(exc)]+=1
            continue
        methods['observed_adaptive'].append(_assess(sc,actions))
    out={}
    for label,rows in methods.items():
        n=len(rows)
        failures=Counter(reason for row in rows for reason in row['reasons'])
        out[label]={'evaluated':n,'matched_proxy':sum(r['matched_kinematic_proxy'] for r in rows),
                    'matched_proxy_fraction':sum(r['matched_kinematic_proxy'] for r in rows)/n if n else None,
                    'nominally_clear_both_count':sum(r['both_clear_constant_velocity'] for r in rows),
                    'nominally_clear_both_fraction':sum(r['both_clear_constant_velocity'] for r in rows)/n if n else None,
                    'matched_and_nominally_clear_count':sum(r['matched_kinematic_proxy'] and r['both_clear_constant_velocity'] for r in rows),
                    'median_candidate_distance':float(np.median([r['d_e'] for r in rows])) if n else None,
                    'matching_failure_count':dict(failures)}
    return {'schema':'CBC_ACTION_DESIGN_OBSERVATION_ONLY_V1',
            'scenes':len(scenes),'policies':out,'adaptive_generation_rejected':dict(rejected),
            'independent_of_reactive_rollouts':True,
            'lawful_route_verified':False,
            'nominal_ego_clear_is_not_reference_ego_safe':True,
            'claim_limitation':'Constant-velocity geometry is only a preliminary screen; neither its success nor failure identifies real contact risk.'}
