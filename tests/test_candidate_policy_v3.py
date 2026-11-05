from caver.sim.traffic import standard_fourcar,prescribed_ego_path
from caver.models.candidate_policy import choose_observation_only_actions
import numpy as np

def test_candidate_generation_observation_only_and_same_intent():
    sc=standard_fourcar()
    acts,info=choose_observation_only_actions(sc)
    assert [a.action_id for a in acts]==['early','late','reference']
    assert acts[0].lane_time==acts[1].lane_time
    assert acts[0].lane_offset==acts[1].lane_offset
    assert 0<=info['ego_candidate_distance_proxy']<=1.8
    assert info['map_law_route_feasibility']=='UNVERIFIED'
    tr=prescribed_ego_path(sc,acts[0])
    assert np.isfinite(tr).all()

def test_observation_candidate_does_not_rely_on_history_future():
    sc=standard_fourcar()
    a,_=choose_observation_only_actions(sc)
    sc.provenance['future_bad']='NOT_EXPOSED'
    for agent in sc.agents:
        agent.history=[[agent.x,agent.y,agent.vx,agent.vy,agent.yaw,1]]
    b,_=choose_observation_only_actions(sc)
    assert a==b
