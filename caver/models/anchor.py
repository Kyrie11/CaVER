"""Candidate features generated from *one* frozen source anchor rollout.

No non-anchor traffic rollout is consulted. Ego candidate motion is computed
analytically from current ego state and proposed controls, not logged future.
This is a predictor reuse experiment; the downstream transport is learned,
not a certified hybrid event-boundary guarantee.
"""
from copy import deepcopy
import math
import numpy as np
from caver.data.schema import Scene
from caver.sim.traffic import Action,prescribed_ego_path


def _candidate_arrival_kernel_shift(graph, candidate_nodes, ego_index, ego_xy,
                                    receiver_xy, sample_dt):
    """Shift ego-to-receiver arrival kernels using a candidate encounter time.

    The source graph stores a receiver delay distribution, while an anchor
    reuse path also needs to account for the fact that a candidate ego pulse
    can enter the receiver's interaction corridor earlier or later.  This
    helper applies one common time shift to the existing discretized kernel.
    It never creates a kernel for an edge that was absent from the anchor
    graph and leaves the anchor kernel unchanged when either encounter is not
    observed in the coarse source trace.
    """
    if not np.isfinite(sample_dt) or sample_dt<=0.:
        return None, {'status':'UNAVAILABLE','reason':'invalid sample interval'}
    candidate_nodes=np.asarray(candidate_nodes,dtype=float)
    if (candidate_nodes.ndim!=3 or candidate_nodes.shape[-1]<2 or
            not 0<=int(ego_index)<candidate_nodes.shape[1] or
            len(receiver_xy)!=candidate_nodes.shape[1]):
        return None, {'status':'UNAVAILABLE','reason':'invalid encounter traces'}
    if any(np.asarray(trace).shape!=(candidate_nodes.shape[0],2) or
           not np.all(np.isfinite(trace)) for trace in receiver_xy):
        return None, {'status':'UNAVAILABLE','reason':'invalid receiver traces'}
    ego_xy=np.asarray(ego_xy,dtype=float)
    if ego_xy.shape!=(candidate_nodes.shape[0],2) or not np.all(np.isfinite(ego_xy)):
        return None, {'status':'UNAVAILABLE','reason':'invalid candidate ego trace'}
    kernels=graph.get('edge_kernels')
    edges=graph.get('edges',[])
    if kernels is None or len(kernels)!=len(edges):
        return None, {'status':'UNAVAILABLE','reason':'edge kernel metadata missing'}

    def first_reachable(ego_trace, recv_trace):
        dx=ego_trace[:,0]-recv_trace[:,0]
        dy=ego_trace[:,1]-recv_trace[:,1]
        hit=np.flatnonzero((dx>=-10.) & (dx<=125.) & (np.abs(dy)<=9.))
        return int(hit[0]) if len(hit) else None

    shifted=[]; shifts={}
    anchor_ego=candidate_nodes[:,int(ego_index),:2]
    for edge_index,(sender,receiver,_nominal_lag) in enumerate(edges):
        kernel=[[int(l),float(w)] for l,w in kernels[edge_index]]
        if sender!=ego_index:
            shifted.append(kernel)
            continue
        anchor_hit=first_reachable(anchor_ego,receiver_xy[receiver])
        candidate_hit=first_reachable(np.asarray(ego_xy,dtype=float),receiver_xy[receiver])
        if anchor_hit is None or candidate_hit is None:
            shifted.append(kernel)
            shifts[str(edge_index)]={'status':'UNCHANGED_NO_COMMON_ENCOUNTER',
                                     'anchor_index':anchor_hit,
                                     'candidate_index':candidate_hit,
                                     'shift_s':0.}
            continue
        delta_s=float(candidate_hit-anchor_hit)*float(sample_dt)
        out=[]
        for lag,weight in kernel:
            out.append([max(1,int(math.ceil(lag+delta_s/sample_dt))),weight])
        # Merge duplicate lags created by discretization and renormalize the
        # original kernel weights so a timing shift cannot change confidence.
        merged={}
        for lag,weight in out:merged[lag]=merged.get(lag,0.)+weight
        mass=sum(merged.values())
        shifted.append([[int(l),float(w/mass)] for l,w in sorted(merged.items())])
        shifts[str(edge_index)]={'status':'SHIFTED','anchor_index':anchor_hit,
                                 'candidate_index':candidate_hit,
                                 'shift_s':delta_s}
    return shifted, {'status':'APPLIED','edges':shifts,
                     'source':'ANCHOR_TRACE_ENCOUNTER_TIME_HEURISTIC'}


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
        anchor_nodes=nodes.copy()
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
        # Preserve the anchor receiver traces and shift only the arrival
        # kernel on ego-originating edges.  This gives the delay-kernel branch
        # candidate-conditioned timing evidence without fabricating a new
        # response trajectory for the mediator.
        receiver_xy=[]
        for actor_index in range(nodes.shape[1]):
            xy=np.empty((nodes.shape[0],2),dtype=float)
            xy[:,0]=anchor_nodes[:,actor_index,0]*100.+ex.x
            xy[:,1]=anchor_nodes[:,actor_index,1]*8.+ex.y
            receiver_xy.append(xy)
        shifted,shift_meta=_candidate_arrival_kernel_shift(
            graph,anchor_nodes,e_idx,ego_tr[indices,:2],receiver_xy,step)
        if shifted is not None:
            graph['edge_kernels']=shifted
            graph['candidate_arrival_shift']=shift_meta
        graph['nodes']=nodes.astype('float32').tolist()
        graph['source_only']=True
        graph['anchor_reuse']=anchor_id
        res['candidate_temporal'][a.action_id]=graph
    return res
