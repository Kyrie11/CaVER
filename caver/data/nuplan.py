"""NuPlan read-only DB inventory plus optional official NuPlanScenario extraction.

DB files under public_set_* are not preprocessed scenario caches. No map-dependent
claim is made without the official devkit and map files.
"""
from pathlib import Path
import sqlite3
import json
from .schema import Agent,Scene

def dbs(root,split=None):
    root=Path(root)
    if split:
        if not split.startswith('public_set_'):raise ValueError('Explicit public_set_* split required')
        root=root/split
    return sorted(root.rglob('*.db')) if root.exists() else []

def inspect(root):
    root=Path(root);out={}
    for x in sorted(root.glob('public_set_*')) if root.exists() else []:
        files=dbs(root,x.name)
        out[x.name]={'files':len(files),'bytes':sum(f.stat().st_size for f in files),'example':str(files[0]) if files else None}
    return out

def inspect_db(db):
    path=Path(db).resolve();c=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:
        tables=[row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        fields={name:[r[1] for r in c.execute('PRAGMA table_info("'+name.replace('"','')+'")')] for name in tables}
        if 'lidar_pc' not in fields:raise ValueError('Expected nuPlan lidar_pc table absent: '+str(path))
        return {'db':str(path),'tables':tables,'columns':fields,'lidar_pc_rows':c.execute('select count(*) from lidar_pc').fetchone()[0]}
    finally:c.close()

def export_scenarios(db,map_root,limit=100,horizon=5.0):
    """Official nuPlan devkit is mandatory for map-aware scenario extraction.

    Caller must supply map DBs and a compatible nuplan-devkit installation.
    Returns (Scene, past/future labels) from API, never SQL schema guesses.
    """
    if not map_root or not Path(map_root).exists():raise FileNotFoundError('nuPlan maps missing; pass --map-root')
    try:
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    except ImportError as exc:
        raise RuntimeError('Official nuplan-devkit required; data DB inventory alone is not a scenario extractor') from exc
    paths=[]
    for i,row in enumerate(get_scenarios_from_db(str(db),None,None,None)):
        if i>=limit:break
        token=row['token'].hex() if isinstance(row['token'],bytes) else str(row['token'])
        t=int(row['timestamp']); map_name=row['map_name']
        sc=NuPlanScenario(data_root=str(Path(db).parent),log_file_load_path=str(db),initial_lidar_token=token,
                          initial_lidar_timestamp=t,scenario_type=row['scenario_type'] or 'unknown',
                          map_root=str(map_root),map_version='nuplan-maps-v1.0',map_name=map_name,
                          scenario_extraction_info=ScenarioExtractionInfo('caver_sampling',horizon,.5),
                          ego_vehicle_parameters=get_pacifica_parameters())
        paths.append(sc)
    return paths

def from_official_scenario(sc, history_horizon=1.0, iteration=0):
    """Planning input via NuPlanScenario read-only history API, never expert future.

    The official get_past_* queries use only sensor samples at or before the
    planning iteration; NuPlan route roadblocks are deliberately NOT extracted
    because the devkit may derive them from future expert states.
    """
    from collections import defaultdict
    if history_horizon <= 0:
        raise ValueError('positive history_horizon required')
    ego = sc.get_ego_state_at_iteration(iteration)
    current = sc.get_tracked_objects_at_iteration(iteration).tracked_objects
    dt = float(sc.database_interval)
    count = max(1, int(round(history_horizon / dt)))
    past_ego = list(sc.get_ego_past_trajectory(iteration, history_horizon, num_samples=count))
    past_agents = list(sc.get_past_tracked_objects(iteration, history_horizon, num_samples=count))
    past_times = list(sc.get_past_timestamps(iteration, history_horizon, num_samples=count))
    if not (len(past_ego) == len(past_agents) == len(past_times)):
        raise ValueError('nuPlan past history API returned misaligned samples')
    # NuPlan API samples may be listed newest-first. Explicit timestamp sorting
    # is required; do not silently reverse based on an assumption.
    pairs = sorted(zip(past_times, past_ego, past_agents), key=lambda x:x[0].time_us)
    ego_hist = []
    agent_hist = defaultdict(list)
    for t, e, objects in pairs:
        if hasattr(e,'time_point') and e.time_point.time_us != t.time_us:
            raise ValueError('nuPlan ego history and scene clock are not aligned')
        c = e.center
        v = e.dynamic_car_state.center_velocity_2d
        ego_hist.append((t.time_us, [float(c.x),float(c.y),float(v.x),float(v.y),float(c.heading),1]))
        for item in objects.tracked_objects:
            if getattr(item.tracked_object_type,'name','') != 'VEHICLE':continue
            vv = item.velocity
            agent_hist[item.track_token].append((t.time_us,
                [float(item.box.center.x),float(item.box.center.y),float(vv.x),float(vv.y),
                 float(item.box.center.heading),1]))
    curr_time = sc.get_time_point(iteration).time_us
    if ego_hist and ego_hist[-1][0] >= curr_time:
        raise ValueError('nuPlan prior observation is not strictly before planning time')
    c=ego.center;v=ego.dynamic_car_state.center_velocity_2d
    current_ego=[float(c.x),float(c.y),float(v.x),float(v.y),float(c.heading),1]
    sampled_times = [t for t,_,_ in pairs] + [sc.get_time_point(iteration)]
    if len({t.time_us for t in sampled_times}) != len(sampled_times):
        raise ValueError('Duplicate nuPlan observation timestamps')
    e=ego.car_footprint.vehicle_parameters
    agents=[Agent('ego','vehicle',*current_ego[:5],float(e.length),float(e.width),
                  history=[frame for _,frame in ego_hist]+[current_ego])]
    time_ids = [t.time_us for t in sampled_times]
    for item in current:
        if getattr(item.tracked_object_type,'name','')!='VEHICLE':continue
        b=item.box;v=item.velocity
        current_frame=[float(b.center.x),float(b.center.y),float(v.x),float(v.y),float(b.center.heading),1]
        past = {t: frame for t,frame in agent_hist[item.track_token]}
        history=[past.get(t,[0.,0.,0.,0.,0.,0]) for t in time_ids[:-1]]+[current_frame]
        agents.append(Agent(item.track_token,'vehicle',*current_frame[:5],float(b.length),
                            float(b.width),history=history))
    return Scene(str(sc.token), 'NUPLAN_OBSERVED_HISTORY_ONLY', dt, 'ego', agents,
                 {'log_name':sc.log_name, 'map_name':getattr(sc,'_map_name',None),
                  'scenario_type':sc.scenario_type, 'planning_iteration':iteration,
                  'planning_time_us':curr_time,'history_steps':len(sampled_times),
                  'future_expert_used':False,'future_trajectory_exposed':False,
                  'overlap_group':'NUPLAN_LOG:'+sc.log_name,
                  'overlap_group_quality':'EXACT_LOG_FILENAME'},
                 map_data={'map_api':'nuplan-devkit','map_name':getattr(sc,'_map_name',None),
                           'route':None, 'route_source':'NOT_GIVEN_AVOID_FUTURE_EXPERT_ROUTE'}).validate()
