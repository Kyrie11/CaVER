"""Finite-horizon, delay-respecting *source-response* graph residual prototype.

This network cannot implement the paper's physically validated event-boundary
certificate. It uses only a declared source reactor's candidate-conditioned
predictions, not logged ground-truth futures or the independent audit reactor.

Graph edges are conservatively screened from CURRENT observed geometry; no
learned importance weight can delete an upstream eligible directed edge.
"""
import math
import numpy as np
import torch
from torch import nn

TRACE_FEATURES=('rel_x','rel_y','speed','acceleration','brake_guard','delay')


def source_temporal_graph(scene, rollouts, dt_sample=.4):
    if not rollouts:
        raise ValueError('No source rollouts for temporal graph')
    ids=[a.agent_id for a in scene.agents]
    actors={a.agent_id:a for a in scene.agents}
    ego=actors[scene.ego_id]
    raw=[]
    # Symmetric aggregation: graph representation must not depend on the
    # arbitrary order of Monte Carlo samples. Computing running means while
    # creating traces would bias all earlier samples and break permutation
    # invariance of the source-world empirical estimate.
    delay_mean={k:float(np.mean([r.get('driver_delay_s',{}).get(k,0.)
                                   for r in rollouts])) for k in ids}
    for r in rollouts:
        if list(r['actor_ids'])!=ids:
            raise ValueError('Source actor order mismatch')
        traj=np.asarray(r['trajectory'],float)
        if traj.ndim!=3 or traj.shape[1:]!=(len(ids),6):
            raise ValueError('Invalid source rollout trajectory')
        if not np.all(np.isfinite(traj)):
            raise ValueError('Nonfinite source rollout trajectory')
        accelerations=r.get('accelerations',{})
        if any(key not in accelerations for key in ids):
            raise ValueError('Source temporal acceleration actor mismatch')
        if any(len(accelerations[key])!=len(traj)-1 for key in ids):
            raise ValueError('Source temporal acceleration length mismatch')
        take=np.arange(0,len(traj),max(1,round(dt_sample/scene.dt)))
        accel=np.zeros((len(traj),len(ids)),float)
        for i,key in enumerate(ids):
            a=np.asarray(accelerations[key],float)
            if not np.all(np.isfinite(a)):
                raise ValueError('Nonfinite source temporal acceleration')
            accel[:-1,i]=a
            accel[-1,i]=a[-1] if len(a) else 0.
        local=np.stack(((traj[take,:,0]-ego.x)/100.,
                        (traj[take,:,1]-ego.y)/8.,
                         traj[take,:,2]/30.,
                         accel[take]/6.,
                         (accel[take]+.6)/6.,
                         np.tile([delay_mean[k] for k in ids],(len(take),1))/1.5),axis=-1)
        raw.append(local)
    nodes=np.mean(raw,axis=0)
    if not np.all(np.isfinite(nodes)):
        raise ValueError('Nonfinite source graph')
    edges=[]
    edge_kernels=[]
    arrival_quantiles={}
    for i,send in enumerate(scene.agents):
        for j,recv in enumerate(scene.agents):
            if i==j or recv.agent_id==scene.ego_id:continue
            dx=send.x-recv.x
            dy=send.y-recv.y
            # A source agent can influence a physically reachable follower;
            # include ego even if an initial lane-change is required.
            if -10. <= dx <= 125. and abs(dy) <= 9.:
                receiver_delays=np.asarray([
                    r.get('driver_delay_s',{}).get(recv.agent_id,delay_mean[recv.agent_id])
                    for r in rollouts],dtype=float)
                if (not np.all(np.isfinite(receiver_delays)) or
                        np.any(receiver_delays < 0.)):
                    raise ValueError('Source driver delays must be finite and nonnegative')
                receiver_delay=float(np.mean(receiver_delays))
                lag=max(1,int(math.ceil(receiver_delay/dt_sample)))
                edges.append([i,j,lag])
                q10,q50,q90=np.quantile(receiver_delays,[.1,.5,.9])
                arrival_quantiles[recv.agent_id]={'q10_s':float(q10),'q50_s':float(q50),'q90_s':float(q90)}
                lags=[max(1,int(math.ceil(float(q)/dt_sample))) for q in (q10,q50,q90)]
                merged={}
                for l,w in zip(lags,(.25,.5,.25)):merged[l]=merged.get(l,0.)+w
                edge_kernels.append([[int(l),float(w)] for l,w in sorted(merged.items())])
    return {'actors':ids,'nodes':nodes.astype('float32').tolist(),
            'edges':edges,'edge_kernels':edge_kernels,
            'arrival_quantiles_s':arrival_quantiles,
            'sample_dt':float(dt_sample),'source_only':True,
            'edge_screen':'INITIAL_REACHABILITY_HEURISTIC_NOT_PHYSICAL_COMPLETENESS',
            'latency_accounting':{
                'behavior_delay_in_source_rollout':True,
                'driver_delay_channel_is_descriptive':True,
                'graph_lag_semantics':'representation_transport_lag_not_callback_latency',
                'online_callback_latency_measured':False,
                'double_counting_guard':'Do not add driver_delay again to the online guard; use residual evidence age.'}}


class DelayedGraphEncoder(nn.Module):
    """Parallel strictly past-shifted message passing, no future-to-past paths."""
    def __init__(self,channels=24,depth=2):
        super().__init__()
        self.input=nn.Linear(len(TRACE_FEATURES),channels)
        self.msg=nn.ModuleList(nn.Linear(channels,channels,bias=False) for _ in range(depth))
        self.update=nn.ModuleList(nn.Sequential(nn.Linear(channels*3,channels),nn.SiLU(),
                                                 nn.Linear(channels,channels)) for _ in range(depth))
        self.norm=nn.ModuleList(nn.LayerNorm(channels) for _ in range(depth))
        self.channels=channels

    def forward(self,nodes,edges,edge_kernels=None):
        if nodes.ndim!=3 or nodes.shape[-1]!=len(TRACE_FEATURES):
            raise ValueError('Graph input must be [T, N, 6]')
        T,N,_=nodes.shape
        h=self.input(nodes)
        for mp,update,norm in zip(self.msg,self.update,self.norm):
            # Every edge uses t-lag, lag>=1. A perturbation in source time t
            # cannot alter graph encodings at any earlier time.
            agg=torch.zeros_like(h)
            # Track the kernel mass that is actually available at each
            # receiver/time location. A fixed full-horizon denominator would
            # dilute messages when only a subset of incoming edges has
            # arrived. The clamp keeps partial kernel mass explicit and
            # preserves a zero message before the first causal arrival.
            counts=torch.zeros((T,N),device=h.device,dtype=h.dtype)
            if edge_kernels is not None and len(edge_kernels)!=len(edges):
                raise ValueError('Edge kernel count does not match edge count')
            for edge_index,(a,b,lag) in enumerate(edges):
                if not (0<=a<N and 0<=b<N and a!=b and lag>=1):
                    raise ValueError('Invalid or noncausal directed edge')
                kernel=edge_kernels[edge_index] if edge_kernels is not None else [[lag,1.]]
                for kernel_lag,kernel_weight in kernel:
                    kernel_lag=int(kernel_lag);kernel_weight=float(kernel_weight)
                    if kernel_lag<1 or kernel_weight<0. or not math.isfinite(kernel_weight):
                        raise ValueError('Invalid or noncausal edge kernel')
                    if kernel_lag<T and kernel_weight>0.:
                        agg[kernel_lag:,b,:]=agg[kernel_lag:,b,:]+kernel_weight*mp(h[:-kernel_lag,a,:])
                        counts[kernel_lag:,b]+=kernel_weight
            agg=agg/counts.clamp_min(1.)[...,None]
            prev=torch.cat((torch.zeros_like(h[:1]),h[:-1]),dim=0)
            h=norm(h+update(torch.cat((h,prev,agg),dim=-1)))
        return h


class GraphCascadeRiskHead(nn.Module):
    """Candidate-conditioned delayed graph correction to physical-anchor risks.

    This is a trainable graph baseline, NOT a verified anchor-reuse oracle.
    """
    def __init__(self,feature_dim,channels=16,event_pool=False,event_local=False,event_window=3,delay_kernel=False):
        super().__init__()
        self.event_pool=bool(event_pool)
        self.event_local=bool(event_local)
        self.event_window=max(1,int(event_window))
        self.delay_kernel=bool(delay_kernel)
        self.graph=DelayedGraphEncoder(channels=channels)
        self.pair=nn.Sequential(nn.Linear(feature_dim+channels*3,64),nn.SiLU(),
                                nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))

    def _encode(self, graph_data, dtype, device):
        nodes=torch.as_tensor(graph_data['nodes'],dtype=dtype,device=device)
        edges=graph_data['edges']
        ids=graph_data['actors']
        h_time=self.graph(nodes,edges,
                          graph_data.get('edge_kernels') if self.delay_kernel else None)
        if self.event_pool:
            # Event-local pooling is an exploratory representation.  It keeps
            # the recent boundary evidence instead of diluting a short braking
            # onset across the full horizon.  It is not a physical certificate.
            event_score=torch.relu(-nodes[...,4])+torch.relu(-nodes[...,3])
            event_score=event_score.mean(dim=1)
            if self.event_local:
                window=min(self.event_window,int(h_time.shape[0]))
                h_sel=h_time[-window:]
                scores=event_score[-window:]
            else:
                h_sel=h_time
                scores=event_score
            weights=1.+4.*scores
            h=(h_sel*weights[:,None,None]).sum(dim=0)/weights.sum().clamp_min(1e-6)
        else:
            h=h_time.mean(dim=0)
        # A joint head receives both global mean and max summaries.  Keeping
        # this tensor separate also makes it possible to audit whether the
        # downstream prediction actually consumed graph evidence.
        summary=torch.cat((h.mean(dim=0),h.max(dim=0).values),dim=-1)
        return h,summary,ids

    def forward(self,x,graph_data,pair_ids,return_summary=False):
        h,summary,ids=self._encode(graph_data,x.dtype,x.device)
        # Shared time causal graph features for both selected pair roles and
        # the scene-wide source response. Pair potential is order-independent.
        globally=h.mean(dim=0)
        vector=[]
        for p in pair_ids:
            i,j=(ids.index(q) for q in p.split('|'))
            vector.append(torch.cat((h[i]+h[j],torch.abs(h[i]-h[j]),globally)))
        z=self.pair(torch.cat((x,torch.stack(vector)),dim=-1))
        source=x[:,9].clamp(.005,.995)
        contact=torch.sigmoid(torch.logit(source)+z[:,0])
        output=torch.stack((contact,x[:,10]+z[:,1],x[:,11]+z[:,2]),dim=-1)
        return (output,summary) if return_summary else output
