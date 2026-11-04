"""Minimal actual CARLA four-actor physical pilot, optional external server.

This is an exploratory dynamic validation with directly controlled actors,
not a recreation of the missing original RIDE fourcar module. On any failure,
actors/sensors are destroyed and world synchronous settings restored.
"""
from pathlib import Path
from collections import deque
import json,math,time

def _next_wp(wp,distance):
    ps=wp.next(distance)
    if not ps:raise RuntimeError('Road waypoint ended while constructing 4-car scene')
    return ps[0]
def _prev_wp(wp,distance):
    ps=wp.previous(distance)
    if not ps:raise RuntimeError('Road waypoint backward unavailable')
    return ps[0]

def _profile_speed(action,t,base=13.):
    return max(0.,base-action.decel*max(0,min(t-action.pulse_begin,action.pulse_end-action.pulse_begin)))

def run_server(host='127.0.0.1',port=2000,town='Town04',out='outputs/carla',seed=10,step=.05,horizon=8.):
    try:import carla
    except ImportError as exc:raise RuntimeError('CARLA Python egg/wheel not visible; source CARLA PythonAPI for installed server') from exc
    from caver.sim.traffic import candidates
    client=carla.Client(host,int(port));client.set_timeout(15.)
    world=client.get_world()
    if world.get_map().name.split('/')[-1]!=town:
        world=client.load_world(town)
    settings=world.get_settings()
    old_sync=settings.synchronous_mode;old_step=settings.fixed_delta_seconds
    bps=world.get_blueprint_library().filter('vehicle.*')
    models=[b for b in bps if b.has_attribute('number_of_wheels') and int(b.get_attribute('number_of_wheels').as_int())==4]
    if len(models)<1:raise RuntimeError('No four-wheeled CARLA vehicle blueprints')
    spawns=world.get_map().get_spawn_points()
    def select():
        # Needs B waypoint with drivable adjacent lane, long empty forward/backward corridor.
        for sp in spawns:
            base=world.get_map().get_waypoint(sp.location,project_to_road=True)
            adjacent=base.get_left_lane() or base.get_right_lane()
            if adjacent is None or int(adjacent.lane_type)!=int(carla.LaneType.Driving):continue
            try:
                A=_next_wp(base,55).transform; B=base.transform;C=_prev_wp(base,25).transform
                E=_next_wp(adjacent,12).transform
            except RuntimeError:continue
            return {'E':E,'A':A,'B':B,'C':C}
        raise RuntimeError('No complete two-lane/longitudinal CARLA spawn corridor; specify different map')
    starts=select();actors=[];sensors=[];results=[]
    cfg=world.get_settings();cfg.synchronous_mode=True;cfg.fixed_delta_seconds=step
    try:
        world.apply_settings(cfg)
        for action in candidates():
            roles={};contacts=[];actor_ids={};buf={}
            try:
                for k,tr in starts.items():
                    bp=models[0];bp.set_attribute('role_name','caver_'+k)
                    transform=carla.Transform(tr.location+carla.Location(z=.25),tr.rotation)
                    a=world.try_spawn_actor(bp,transform)
                    if a is None:raise RuntimeError('Failed to spawn '+k+' (road occupied)')
                    actors.append(a);roles[k]=a;actor_ids[a.id]=k
                for k,a in roles.items():
                    bp=world.get_blueprint_library().find('sensor.other.collision')
                    sensor=world.spawn_actor(bp,carla.Transform(),attach_to=a);sensors.append(sensor)
                    sensor.listen(lambda ev,k=k:contacts.append({'t':float(ev.timestamp),'a':k,'other_id':ev.other_actor.id}))
                world.tick()
                # Per-role delayed observations: this buffer includes location and speed,
                # not solely a brake command. CARLA physical acceleration is measured.
                traces=[];delays={'A':.2,'B':.45,'C':.75}
                buf={k:deque(maxlen=max(1,round(delays[k]/step))+1) for k in delays}
                for it in range(round(horizon/step)):
                    t=it*step;frames={}
                    for k,a in roles.items():
                        tr=a.get_transform();v=a.get_velocity()
                        frames[k]={'x':tr.location.x,'y':tr.location.y,'z':tr.location.z,
                                   'v':math.sqrt(v.x*v.x+v.y*v.y),'yaw_deg':tr.rotation.yaw}
                    for k in delays:buf[k].append({j:dict(v) for j,v in frames.items()})
                    def control_speed(a,target,lead=None,distance=None):
                        measured=a.get_velocity();v=math.hypot(measured.x,measured.y)
                        err=target-v
                        if distance is not None and lead is not None:
                            safe=max(4.,1.15*v)
                            if distance<safe:err=min(err,-(safe-distance)*.6)
                        return carla.VehicleControl(throttle=float(max(0,min(.6,err*.18))),
                                                    brake=float(max(0,min(1.,-err*.24))),steer=0.)
                    # A holds its speed; followers respond to stale physical observations.
                    roles['A'].apply_control(control_speed(roles['A'],13.))
                    for k in ('B','C'):
                        hist=buf[k][0]
                        ego_wp=roles[k].get_transform()
                        lat=math.radians(ego_wp.rotation.yaw)
                        ex=frames[k]['x'];ey=frames[k]['y']
                        lead=[]
                        for m,v in hist.items():
                            if m==k:continue
                            delta_x=v['x']-hist[k]['x'];delta_y=v['y']-hist[k]['y']
                            long=delta_x*math.cos(lat)+delta_y*math.sin(lat)
                            lateral=abs(-delta_x*math.sin(lat)+delta_y*math.cos(lat))
                            if long>0 and lateral<2.2:lead.append(long-4.5)
                        gap=min(lead) if lead else None
                        roles[k].apply_control(control_speed(roles[k],15. if k=='B' else 15.5,
                                                distance=gap,lead=bool(lead)))
                    # Ego: pure pursuit towards adjoining lane centerline with planned speed.
                    E=roles['E'];wp=world.get_map().get_waypoint(E.get_location())
                    target=_next_wp(wp,8.)
                    to_lane=world.get_map().get_waypoint(starts['B'].location)
                    lane=target.get_right_lane() or target.get_left_lane()
                    if lane is not None and lane.road_id==to_lane.road_id and lane.lane_id==to_lane.lane_id:
                        aim=lane.transform.location
                    else:aim=target.transform.location
                    rot=math.radians(E.get_transform().rotation.yaw)
                    x=aim.x-E.get_location().x;y=aim.y-E.get_location().y
                    cross=-x*math.sin(rot)+y*math.cos(rot)
                    speed_ctl=control_speed(E,_profile_speed(action,t))
                    speed_ctl.steer=float(max(-.5,min(.5,cross/12.)))
                    E.apply_control(speed_ctl)
                    world.tick();traces.append({'t':t,'state':frames})
                # Resolve collisions by the original actor role mapping; contact sensors
                # can count the same event twice; deduplicate pair role identities.
                pairs=set()
                for c in contacts:
                    other=actor_ids.get(c['other_id'])
                    if other:pairs.add('|'.join(sorted((c['a'],other))))
                results.append({'action':action.action_id,'world_map':world.get_map().name,
                                'contact_pairs':sorted(pairs),'ego_collision':any('E' in p.split('|') for p in pairs),
                                'downstream_collision':any('E' not in p.split('|') for p in pairs),
                                'traces':traces,'actual_carla_physics':True,'seed':seed})
            finally:
                for sensor in sensors:
                    sensor.stop();sensor.destroy()
                sensors.clear()
                for a in actors:a.destroy()
                actors.clear()
                world.tick()
    finally:
        cfg=world.get_settings();cfg.synchronous_mode=old_sync;cfg.fixed_delta_seconds=old_step
        world.apply_settings(cfg)
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    (p/'carla_rollouts.json').write_text(json.dumps(results,indent=2))
    return {'output':str(p/'carla_rollouts.json'),'actions':len(results),'verified':'CARLA_SERVER_EXECUTED',
            'note':'No guarantee of matched dynamic initialization beyond spawn state; inspect traces and contacts'}
