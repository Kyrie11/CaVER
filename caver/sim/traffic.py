"""Heterogeneous delayed-following surrogate reactor for *controlled experiments*.

Does not stand in for SMART, CARLA or true human driving. Counterfactual worlds
share sampled latent characteristics by an explicit STRUCTURAL coupling choice.
"""
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional
import math
import numpy as np
from .geometry import trajectory_clearance, stop_slack
from caver.data.schema import Agent, Scene

@dataclass(frozen=True)
class Driver:
    delay: float=0.45
    max_brake: float=4.6
    desired_headway: float=1.2
    max_accel: float=1.2
    comfortable_brake: float=2.8
    min_gap: float=2.0

@dataclass(frozen=True)
class Action:
    action_id: str
    pulse_begin: float
    pulse_end: float
    decel: float
    lane_offset: float=3.5
    lane_time: float=3.0

@dataclass
class World:
    driver: dict[str,Driver]
    seed: int
    policy_family: str='delayed_idm'


def sample_world(scene,seed=0,family='delayed_idm'):
    rng=np.random.default_rng(seed)
    params={}
    for agent in scene.agents:
        if agent.agent_id==scene.ego_id:continue
        if family in ('delayed_idm','learned_logged'):
            d=Driver(delay=float(rng.uniform(.24,.95)),max_brake=float(rng.uniform(2.8,6.0)),
                desired_headway=float(rng.uniform(.8,1.9)),comfortable_brake=float(rng.uniform(1.7,3.5)))
        elif family=='conservative':
            d=Driver(delay=float(rng.uniform(.20,.55)),max_brake=float(rng.uniform(4.,7.)),
                desired_headway=float(rng.uniform(1.5,2.3)),comfortable_brake=3.)
        elif family=='late_brake':
            d=Driver(delay=float(rng.uniform(.65,1.1)),max_brake=float(rng.uniform(2.4,4.5)),
                desired_headway=float(rng.uniform(.7,1.3)),comfortable_brake=2.3)
        else: raise ValueError('Unknown reactor family '+family)
        params[agent.agent_id]=d
    return World(params,seed,family)

def standard_fourcar(offset=0., speed=16.):
    """Synthetic fixture: ego on adjacent lane, A-B-C in target lane.
    Tunable input fixture, not an observed RQ1/RQ2 positive finding.
    """
    return Scene('synthetic-fourcar','SYNTHETIC_ONLY',.1,'E',[
        Agent('E','vehicle',17.,-3.5,speed-2.,0,0,4.5,1.8),
        Agent('A','vehicle',62.,0.,15.,0,0,4.5,1.8),
        Agent('B','vehicle',5.+offset,0.,speed,0,0,4.5,1.8),
        Agent('C','vehicle',-15.+offset,0.,speed+.5,0,0,4.5,1.8)],
        {'synthetic':True,'rules':'delayed-following, no real-world frequency inference'}).validate()

def candidates():
    return [Action('early',.6,1.6,1.8),Action('late',1.8,2.8,1.8),Action('reference',9.,10.,0.)]

def _quintic(q):
    q=max(0.,min(1.,q));return 10*q**3-15*q**4+6*q**5

def _idmdriver(v,front_gap,relative_speed,d):
    if front_gap is None:return np.clip(d.max_accel*(1-(v/19.)**4),-d.max_brake,d.max_accel)
    desired=d.min_gap+max(0.,v*d.desired_headway+v*relative_speed/(2*math.sqrt(max(.1,d.max_accel*d.comfortable_brake))))
    raw=d.max_accel*(1-(v/19.)**4-(desired/max(.2,front_gap))**2)
    return float(np.clip(raw,-d.max_brake,d.max_accel))

def rollout(scene:Scene, action:Action, world:World, horizon=8., dt=.1, override_states=None, response_net=None):
    """All other agents *react* to delayed observations, not logged future.

    Ego is prescribed a dynamically integrated action. Discrete-time collision
    labels are evaluated with substeps and separately carry diagnostic status.
    """
    override_states=override_states or {}
    if world.policy_family=='learned_logged' and response_net is None:
        raise ValueError('learned_logged family requires --policy-checkpoint trained from real histories')
    n=int(round(horizon/dt)); ids=[a.agent_id for a in scene.agents]; ego=ids.index(scene.ego_id)
    state=np.array([[a.x,a.y,math.hypot(a.vx,a.vy),a.yaw,a.length,a.width] for a in scene.agents],float)
    initial_y=state[ego,1];target_y=initial_y+action.lane_offset
    track=np.empty((n+1,len(ids),6),float);track[0]=state
    def starting_buffer(driver_id):
        steps=max(2,int(math.ceil(world.driver[driver_id].delay/dt))+2)
        tracks=[state.copy() for _ in range(steps)]
        # Populate the previous delay window with genuine observations only.
        # Missing/invalid historical snapshots are never read as zero-state.
        for offset in range(1,steps):
            historical=state.copy()
            for j,agent in enumerate(scene.agents):
                if not agent.history or len(agent.history)<=offset:continue
                h=agent.history[-1-offset]
                if len(h)!=6 or not h[5]:continue
                historical[j,:4]=[float(h[0]),float(h[1]),
                                   math.hypot(float(h[2]),float(h[3])),float(h[4])]
            tracks[-1-offset]=historical
        return deque(tracks,maxlen=steps)
    hist=[starting_buffer(k) if k!=scene.ego_id else None for k in ids]
    onset={k:None for k in ids if k!=scene.ego_id};accel_records={k:[] for k in ids};
    for tix in range(n):
        t=tix*dt
        nxt=state.copy()
        ego_acc=-action.decel if action.pulse_begin<=t<action.pulse_end else 0.
        # prescribed ego path: integrated speed + smoothly changing lane coordinate
        nxt[ego,2]=max(0.,state[ego,2]+ego_acc*dt)
        nxt[ego,0]=state[ego,0]+state[ego,2]*dt+.5*ego_acc*dt*dt
        nxt[ego,1]=initial_y+(target_y-initial_y)*_quintic(min(t+dt,action.lane_time)/action.lane_time)
        nxt[ego,3]=math.atan2((nxt[ego,1]-state[ego,1])/dt,max(.1,nxt[ego,2]))
        accel_records[scene.ego_id].append(ego_acc)
        for idx,ident in enumerate(ids):
            if idx==ego:continue
            if ident in override_states:
                forced=np.asarray(override_states[ident],float)
                if forced.shape!=(n+1,6):raise ValueError('Forced mediator trace shape mismatch')
                nxt[idx]=forced[tix+1]
                accel_records[ident].append((nxt[idx,2]-state[idx,2])/dt)
                continue
            d=world.driver[ident]
            prev=hist[idx][-min(len(hist[idx]),int(math.ceil(d.delay/dt))+1)]
            # choose closest lead aligned with current lane at DELAYED observation time
            valid=[]
            for j in range(len(ids)):
                if j==idx:continue
                ahead=prev[j,0]-prev[idx,0]-.5*(prev[j,4]+prev[idx,4])
                if ahead>0 and abs(prev[j,1]-prev[idx,1])<(prev[j,5]+prev[idx,5])/2+.35:
                    valid.append((ahead,prev[idx,2]-prev[j,2],j))
            if valid:
                gap,rel,frontidx=min(valid,key=lambda x:x[0]); acc=_idmdriver(prev[idx,2],gap,rel,d)
            else:
                gap,rel=None,0.
                acc=_idmdriver(prev[idx,2],None,0,d)
            if world.policy_family=='learned_logged':
                # Observationally fitted acceleration is evaluated on *delayed*
                # joint state. This is a hypothesis reactor, not a true causal law.
                import torch
                x=torch.tensor([[prev[idx,2]/25.,min(70.,max(0.,gap if gap is not None else 70.))/70.,
                                  rel/15.,math.sin(prev[idx,3]),math.cos(prev[idx,3])]],dtype=torch.float32)
                with torch.no_grad():logged_acc=float(response_net(x).item())
                acc=float(np.clip(logged_acc,-d.max_brake,d.max_accel))
            nxt[idx,2]=max(0.,state[idx,2]+acc*dt)
            nxt[idx,0]=state[idx,0]+state[idx,2]*dt+.5*acc*dt*dt
            accel_records[ident].append(acc)
            if onset[ident] is None and acc<-.6:onset[ident]=t
        state=nxt
        track[tix+1]=state
        for idx,ident in enumerate(ids):
            if idx!=ego:hist[idx].append(state.copy())
    # Geometry API expects [x,y,yaw,length,width]; simulator stores
    # [x,y,SPEED,yaw,length,width]. Passing the full state used to treat speed
    # as yaw, yaw as length and length as width: a major false contact bug.
    # Vectorize the shared interpolation once for all pairs. Swept sampling
    # remains a discrete approximation, NOT a continuous collision certificate.
    fractions=np.array([0.,.25,.5,.75],dtype=float)
    inter=(track[:-1,None,:,:]*(1.-fractions)[None,:,None,None]
          +track[1:,None,:,:]*fractions[None,:,None,None]).reshape(n*4,len(ids),6)
    inter=np.concatenate([inter,track[-1:]],axis=0)
    geo=inter[:,:,[0,1,3,4,5]]
    contact={}; min_clearance={}; slack={}
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            ia,ib=ids[i],ids[j];p='|'.join(sorted((ia,ib)))
            dist=trajectory_clearance(geo[:,i,:],geo[:,j,:])
            contact[p]=int(dist<=0);min_clearance[p]=float(dist)
            if ia!=scene.ego_id and ib!=scene.ego_id:
                rear,front=(i,j) if track[0,i,0]<track[0,j,0] else (j,i)
                if abs(track[0,i,1]-track[0,j,1])<1.5:
                    dr=world.driver[ids[rear]]
                    gap=track[0,front,0]-track[0,rear,0]-.5*(track[0,front,4]+track[0,rear,4])
                    slack[p]=float(stop_slack(gap,track[0,rear,2],track[0,front,2],dr.delay,dr.max_accel,dr.max_brake,world.driver[ids[front]].max_brake))
    # Event-conditioned policy slack: only evaluate at first observed leader
    # braking conflict, with validated longitudinal alignment and gap.
    # A missing event is INVALID, never replaced by a convenient constant.
    event_slack={};slack_valid={};detect_time={}
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            ia,ib=ids[i],ids[j]
            if ia==scene.ego_id or ib==scene.ego_id:continue
            p='|'.join(sorted((ia,ib)))
            rear,front=(i,j) if track[0,i,0]<track[0,j,0] else (j,i)
            lead_acc=accel_records[ids[front]]
            onset_idx=next((k for k,acc in enumerate(lead_acc) if acc<-.6),None)
            slack_valid[p]=False
            if onset_idx is None:continue
            st=track[onset_idx]
            if (st[front,0]<=st[rear,0] or abs(st[front,1]-st[rear,1])>=1.5
                    or abs(st[front,3]-st[rear,3])>=.25):continue
            gap=st[front,0]-st[rear,0]-.5*(st[front,4]+st[rear,4])
            if gap<0:continue
            rear_pol=world.driver[ids[rear]];front_pol=world.driver[ids[front]]
            event_slack[p]=float(stop_slack(gap,st[rear,2],st[front,2],rear_pol.delay,
                                             rear_pol.max_accel,rear_pol.max_brake,front_pol.max_brake))
            slack_valid[p]=True;detect_time[p]=onset_idx*dt
    e_contact=int(any(v and scene.ego_id in pair.split('|') for pair,v in contact.items()))
    downstream=int(any(v and scene.ego_id not in pair.split('|') for pair,v in contact.items()))
    return {'scene_id':scene.scene_id,'action_id':action.action_id,'world_seed':world.seed,'family':world.policy_family,
            'actor_ids':ids,'trajectory':track,'contact':contact,'min_clearance':min_clearance,
            'stop_slack_initial':slack,'stop_slack_event':event_slack,
            'stop_slack_valid':slack_valid,'brake_detection_time':detect_time,'response_onset':onset,'accelerations':accel_records,
            'ego_collision':e_contact,'downstream_collision':downstream,'progress':float(track[-1,ego,0]-track[0,ego,0]),
            'is_surrogate':True,'mediator_forced':sorted(override_states),'label_quality':'SURROGATE_NUMERICAL_4_SUBSTEPS'}
