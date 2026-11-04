"""Waymax dataloader bridge and native-environment interface preflight.

The native Waymax environment must be supplied ego-controlled actions and
reactive agent controllers; merely loading a WOMD state is NOT a closed-loop
reactive benchmark. This module deliberately calls its operation 'read'.
"""
from pathlib import Path
import dataclasses
import numpy as np
from caver.data.womd import find_shards
from caver.data.schema import Scene,Agent

def load_waymax_states(root,split='training',limit=2,max_objects=32):
    folder,shards=find_shards(root,'tf_example',split)
    if not shards:raise FileNotFoundError(f'Waymax needs WOMD tf_example/{split} uncompressed TFRecord shards under {folder}')
    try:
        from waymax import config,dataloader
    except ImportError as e:raise RuntimeError('waymax and its TensorFlow/JAX dependencies not available') from e
    # Inspect the installed DatasetConfig rather than hardcoding WOD_1_1_0_PATH.
    cfg=config.DatasetConfig(path=str(shards[0]),max_num_objects=max_objects,repeat=1,
                             shuffle_buffer_size=1,num_shards=1,deterministic=True)
    it=dataloader.simulator_state_generator(config=cfg)
    for idx,state in zip(range(limit),it):
        yield state,str(shards[0]),idx

def normalize_waymax_state(state,source_shard,index=0):
    """Uses official datatypes fields; raises if installed API is incompatible."""
    traj=state.log_trajectory
    meta=state.object_metadata
    t=int(np.asarray(state.timestep))
    if t<0 or t>=np.asarray(traj.x).shape[-1]:
        raise ValueError('Waymax planning timestep out of observed range')
    x=np.asarray(traj.x)[:,t];y=np.asarray(traj.y)[:,t]
    h=np.asarray(traj.yaw)[:,t];vx=np.asarray(traj.vel_x)[:,t];vy=np.asarray(traj.vel_y)[:,t]
    l=np.asarray(traj.length)[:,t];w=np.asarray(traj.width)[:,t];valid=np.asarray(traj.valid)[:,t]
    is_sdc=np.asarray(meta.is_sdc)
    ego_idx=np.flatnonzero(is_sdc)
    if len(ego_idx)!=1:raise ValueError('Waymax state has no unique SDC')
    agents=[]
    for i in range(len(x)):
        if not valid[i] or not np.isfinite(x[i]) or l[i]<=0 or w[i]<=0:continue
        # Do not pass the rest of state.log_trajectory (> t) to CaVER.
        hist=[]
        for hix in range(max(0,t-10),t+1):
            ok=bool(np.asarray(traj.valid)[i,hix])
            if ok:
                hist.append([float(np.asarray(traj.x)[i,hix]),float(np.asarray(traj.y)[i,hix]),
                             float(np.asarray(traj.vel_x)[i,hix]),float(np.asarray(traj.vel_y)[i,hix]),
                             float(np.asarray(traj.yaw)[i,hix]),1])
            else: hist.append([0.,0.,0.,0.,0.,0])
        agents.append(Agent(str(i),'vehicle',float(x[i]),float(y[i]),float(vx[i]),float(vy[i]),
                            float(h[i]),float(l[i]),float(w[i]),history=hist))
    sid=getattr(state,'scenario_id',None)
    identity=(str(np.asarray(sid).tolist()) if sid is not None else str(index))
    return Scene('waymax:'+identity+':'+str(index),'WAYMAX_LOADED_WOMD',.1,str(ego_idx[0]),agents,
                 {'source_shard':source_shard,'native_waymax_closed_loop':False,
                  'state_format':'SimulatorState.log_trajectory','history_only':True,
                  'future_trajectory_exposed':False,'read_scope':'FIRST_SHARD_PREFLIGHT_ONLY'}).validate()
