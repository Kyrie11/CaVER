"""CBC physical-anchor risk head and differentiable directed message encoder.

This is a *trainable reference model* not an official SMART checkpoint. The
source rollout estimates and correction are separately logged and calibrated.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch import nn

FEATURES=('initial_gap','initial_lateral','follower_speed','leader_speed',
 'ego_to_follower','ego_to_leader','pulse_begin','pulse_end','decel',
 'source_pair_contact','source_pair_clearance','source_event_stop_slack','source_slack_valid_fraction',
 'source_follower_brake_onset','source_leader_brake_onset',
 'source_downstream_union','source_ego_contact','path_hops','source_samples')


def pair_features(scene,action,rollouts,pair):
    if not rollouts:raise ValueError('Need source-family rollouts before constructing CBC evidence')
    ids=pair.split('|'); lookup={a.agent_id:a for a in scene.agents}
    aa,bb=lookup[ids[0]],lookup[ids[1]]
    follower,leader=(aa,bb) if aa.x<bb.x else (bb,aa)
    ego=lookup[scene.ego_id]
    source_contact=float(np.mean([r['contact'].get(pair,0) for r in rollouts]))
    minclear=float(np.mean([r['min_clearance'].get(pair,50.) for r in rollouts]))
    valid_slacks=[r['stop_slack_event'][pair] for r in rollouts
                  if r.get('stop_slack_valid',{}).get(pair,False)]
    slack=float(np.mean(valid_slacks)) if valid_slacks else 0.
    valid_frac=len(valid_slacks)/len(rollouts)
    onset1=[r['response_onset'].get(follower.agent_id) for r in rollouts]
    onset2=[r['response_onset'].get(leader.agent_id) for r in rollouts]
    def onset_avg(x):return float(np.mean([v if v is not None else 9. for v in x]))
    # Path hops = ego->leader->follower, when relative geometry allows it.
    hops=2. if 0<leader.x-ego.x<50 and leader.x>follower.x else 1.
    v=[leader.x-follower.x-.5*(leader.length+follower.length),abs(leader.y-follower.y),
       math.hypot(follower.vx,follower.vy),math.hypot(leader.vx,leader.vy),
       follower.x-ego.x,leader.x-ego.x,action.pulse_begin,action.pulse_end,action.decel,
       source_contact,minclear,slack,valid_frac,onset_avg(onset1),onset_avg(onset2),
       float(np.mean([r['downstream_collision'] for r in rollouts])),
       float(np.mean([r['ego_collision'] for r in rollouts])),hops,float(len(rollouts))]
    return np.asarray(v,dtype=np.float32)

def all_downstream_pairs(scene):
    ids=sorted(a.agent_id for a in scene.agents if a.agent_id!=scene.ego_id)
    return [x+'|'+y for i,x in enumerate(ids) for y in ids[i+1:]]

class CascadeRiskHead(nn.Module):
    """Physics-anchored event/boundary correction; no raw risk-score only head."""
    def __init__(self,width=96):
        super().__init__()
        self.feature_norm=nn.LayerNorm(len(FEATURES))
        self.encoder=nn.Sequential(nn.Linear(len(FEATURES),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.output=nn.Linear(width,3) # contact logit correction, margin residual, stopping-slack residual

    def forward(self,x):
        z=self.output(self.encoder(self.feature_norm(x)))
        # Baseline probability is source Monte Carlo contact. Avoid infinite logits.
        source=torch.clamp(x[...,FEATURES.index('source_pair_contact')],.005,.995)
        base_logit=torch.logit(source)
        probability=torch.sigmoid(base_logit+z[...,0])
        clearance=x[...,FEATURES.index('source_pair_clearance')]+z[...,1]
        stopping=x[...,FEATURES.index('source_event_stop_slack')]+z[...,2]
        return torch.stack((probability,clearance,stopping),dim=-1)

class JointRiskHead(nn.Module):
    """Joint-rollout ego/third-party-union risk, NOT sum of pair marginals.

    Each observation is a variable-length non-ego pair collection for one
    (scene, candidate) with an action-independent screened pair set. Source
    ego/union means are physics anchors; gradients learn reference reactor
    corrections from independently sampled *joint* rollouts.
    """
    def __init__(self, width=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.LayerNorm(len(FEATURES)),nn.Linear(len(FEATURES),width),
                                   nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.decoder=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,2))

    def forward(self, x):
        if x.ndim!=2 or x.shape[0]<1 or x.shape[-1]!=len(FEATURES):
            raise ValueError('Expected [downstream_pairs,features] for one candidate')
        z=self.encoder(x)
        correction=self.decoder(torch.cat((z.mean(0),z.max(0).values),dim=-1))
        idx_ego=FEATURES.index('source_ego_contact')
        idx_down=FEATURES.index('source_downstream_union')
        anchor=torch.stack((x[:,idx_ego].mean(),x[:,idx_down].mean())).clamp(.005,.995)
        return torch.sigmoid(torch.logit(anchor)+correction)


def predict_joint(model, scene_pair_features):
    """Output joint ego, union source risk, optionally corrected by CBC.

    Algebraic consistency projection is a conservatively declared model
    postprocessing step; calibration always uses the *postprocessed* output.
    """
    a=np.asarray(scene_pair_features,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=len(FEATURES):
        raise ValueError('Invalid full candidate pair feature matrix')
    if model is not None and hasattr(model,'joint'):
        with torch.no_grad():
            result=model.joint(torch.tensor(a,dtype=torch.float32)).numpy()
        ego,down=map(float,result)
    else:
        ego=float(np.mean(a[:,FEATURES.index('source_ego_contact')]))
        down=float(np.mean(a[:,FEATURES.index('source_downstream_union')]))
    return {'ego':ego,'down':down,'scope':'JOINT_RESPONSE_EVENT_NOT_SUM_PAIRS',
            'model':'CBC_joint_head' if model is not None and hasattr(model,'joint') else 'source_joint_rollout'}


def coherent_project(pred, actions, pairs):
    """Minimal project C_down(a) into the necessary [max C_p,min(1,sum C_p)].

    Every calibration and evaluation must use the same projection; it is NOT
    an independent proof that the predictions match the reference reactor.
    """
    result=dict(pred)
    for a in actions:
        if not pairs:result[a+'/down']=0.;continue
        vals=[float(pred[a+'/'+p]) for p in pairs]
        lo=max(vals);hi=min(1.,sum(vals))
        result[a+'/down']=float(np.clip(pred[a+'/down'],lo,hi))
    return result

def training_loss(pred,target):
    """Targets [reference contact mean, clearance, event slack, event-valid mask].

    An absent event does not contribute to policy-slack supervision. The
    0 supplied for invalid targets is a storage placeholder, never a label.
    """
    if target.shape[-1]!=4:
        raise ValueError('CBC targets require explicit event-slack validity masks')
    risk=torch.nn.functional.binary_cross_entropy(pred[...,0].clamp(1e-6,1-1e-6),target[...,0])
    clearance=torch.nn.functional.smooth_l1_loss(pred[...,1],target[...,1],beta=2)
    valid=target[...,3].clamp(0,1)
    residual=torch.nn.functional.smooth_l1_loss(pred[...,2],target[...,2],beta=3,reduction='none')
    slack=(residual*valid).sum()/valid.sum().clamp_min(1.)
    return risk+.08*clearance+.05*slack

def source_predict(features):
    a=np.asarray(features)
    return {'p':float(a[9]),'margin':float(a[10]),'slack':float(a[11]),'slack_valid':bool(a[12]),'model':'source_rollout'}

def predict(model,features):
    if model is None:return source_predict(features)
    with torch.no_grad():out=model(torch.tensor(np.asarray(features)[None,:],dtype=torch.float32))[0].cpu().numpy()
    return {'p':float(out[0]),'margin':float(out[1]),'slack':float(out[2]),'slack_valid':bool(np.asarray(features)[12]),'model':'CBC_physics_anchored'}

def save(model,path,meta=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'weights':model.state_dict(),'features':FEATURES,'meta':meta or {},
                'has_joint_head':hasattr(model,'joint')},path)

def load(path):
    try:record=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:record=torch.load(path,map_location='cpu')
    if tuple(record['features'])!=FEATURES:raise ValueError('CBC feature schema incompatible with checkpoint')
    m=CascadeRiskHead()
    if record.get('has_joint_head'):
        m.joint=JointRiskHead()
    m.load_state_dict(record['weights']);m.eval()
    return m
