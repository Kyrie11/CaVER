from types import SimpleNamespace as NS
from caver.data.nuplan import from_official_scenario

def fake_ego(x):
    return NS(center=NS(x=x,y=0.,heading=0.),
              dynamic_car_state=NS(center_velocity_2d=NS(x=3.,y=0.)),
              car_footprint=NS(vehicle_parameters=NS(length=4.,width=2.)))

def fake_agent(x):
    return NS(track_token='veh',tracked_object_type=NS(name='VEHICLE'),
              box=NS(center=NS(x=x,y=1.,heading=0.),length=4.,width=2.),
              velocity=NS(x=3.,y=0.))

class API:
    token='token1';database_interval=.1;log_name='log1';map_name='sg';scenario_type='lane_change'
    def get_time_point(self,i):return NS(time_us=1_000_000)
    def get_ego_state_at_iteration(self,i):return fake_ego(10)
    def get_tracked_objects_at_iteration(self,i):return NS(tracked_objects=[fake_agent(9)])
    def get_ego_past_trajectory(self,*args,**kwargs):return iter([fake_ego(9),fake_ego(8)])
    def get_past_timestamps(self,*args,**kwargs):return iter([NS(time_us=900_000),NS(time_us=800_000)])
    def get_past_tracked_objects(self,*args,**kwargs):return iter([NS(tracked_objects=[fake_agent(8)]),NS(tracked_objects=[fake_agent(7)])])
    def get_ego_future_trajectory(self,*args,**kwargs):raise AssertionError('FUTURE LEAK')

def test_nuplan_history_sorted_no_future():
    s=from_official_scenario(API())
    assert s.agents[0].history[0][0]==8
    assert s.agents[0].history[-1][0]==10
    assert s.agents[1].history[0][0]==7
    assert not s.provenance['future_expert_used']
