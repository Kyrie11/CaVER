"""Candidate features generated from *one* frozen source anchor rollout.

No non-anchor traffic rollout is consulted. Ego candidate motion is computed
analytically from current ego state and proposed controls, not logged future.
This is a predictor reuse experiment; the downstream transport is learned,
not a certified hybrid event-boundary guarantee.
"""
from copy import deepcopy
import numpy as np
from caver.data.schema import Scene
from caver.sim.traffic import Action,prescribed_ego_path


def build_anchor_view(row,anchor_id='early'):
    if anchor_id not in row['temporal']:
        raise ValueError('No source anchor temporal trace')
    if not any(a['action_id']==anchor_id for a in row['actions']):
        raise ValueError('Anchor absent from candidate set')
    s=Scene.from_json(row['scene'])
    trace=row['temporal'][anchor_id]
    e_idx=trace['actors'].index(s.ego_id)
    actions=[Action(**a) for a in row['actions']]
    res={'anchor_id':anchor_id,'source':'ONE_ANCHOR_FULL_RESPONSE_PLUS_EGO_CONTROLS_ONLY',
         'candidate_features':{},'candidate_temporal':{}}
    dt=s.dt; step=float(trace['sample_dt']);
    for a in actions:
        # `source` pair-contact/margin/onset features intentionally all come
        # from ONE anchor; action controls are the only varying pair inputs.
        for p in row['pairs']:
            base=np.array(row['diagnostics'][f'{anchor_id}/{p}']['features'],dtype=float)
            base[6:9]=[a.pulse_begin,a.pulse_end,a.decel]
            res['candidate_features'][f'{a.action_id}/{p}']=base.astype('float32').tolist()
        graph=deepcopy(trace)
        nodes=np.asarray(graph['nodes'],dtype=float)
        if nodes.ndim!=3 or nodes.shape[1]!=len(graph['actors']):
            raise ValueError('Invalid anchor-only temporal graph')
        horizon=(nodes.shape[0]-1)*step
        steps=int(round(horizon/dt))
        ego_tr=prescribed_ego_path(s,a,horizon=steps*dt,dt=dt)
        indices=np.clip(np.rint(np.arange(nodes.shape[0])*step/dt).astype(int),0,len(ego_tr)-1)
        ex=next(v for v in s.agents if v.agent_id==s.ego_id)
        ego_acc=np.diff(ego_tr[:,2])/dt
        sampled_acc=ego_acc[np.clip(indices,0,len(ego_acc)-1)]
        nodes[:,e_idx,0]=(ego_tr[indices,0]-ex.x)/100.
        nodes[:,e_idx,1]=(ego_tr[indices,1]-ex.y)/8.
        nodes[:,e_idx,2]=ego_tr[indices,2]/30.
        nodes[:,e_idx,3]=sampled_acc/6.
        nodes[:,e_idx,4]=(sampled_acc+.6)/6.
        nodes[:,e_idx,5]=0.
        graph['nodes']=nodes.astype('float32').tolist()
        graph['source_only']=True
        graph['anchor_reuse']=anchor_id
        res['candidate_temporal'][a.action_id]=graph
    return res
