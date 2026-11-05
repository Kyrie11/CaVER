"""Map must remain in the same normalized observation frame as agents."""
from caver.data.schema import Scene, Agent
from caver.data.corridor import corridor_subset


def _agent(name,x,y):
    return Agent(name,'vehicle',x,y,12.,0.,0.,4.5,1.8,
                 history=[[x-1.2,y,12.,0.,0.,1],[x,y,12.,0.,0.,1]])


def test_corridor_keeps_observed_map_in_local_frame_without_future():
    sc=Scene('s1','WOMD_SCENARIO_PROTO',.1,'E',[
        _agent('E',0,0),_agent('A',42,3.5),_agent('B',4,3.5),_agent('C',-20,3.5)],
        {'history_only':True,'future_trajectory_exposed':False},
        {'polylines':[{'kind':'lane','feature_id':123,'xy':[[0.,3.5],[20.,3.5]]}],
         'observed_signals':[{'step':2,'lane_signals':[{'lane_id':123,'state':1}]}]})
    out,reason=corridor_subset(sc)
    assert reason is None
    assert out is not None
    assert out.provenance['map_preserved']
    assert out.provenance['corridor_filter']=='GEOMETRIC_NOT_MAP_CERTIFIED'
    assert out.map_data['polylines'][0]['xy']==[[0.,3.5],[20.,3.5]]
    assert out.map_data['observed_signals'][0]['step']==2
    assert out.map_data['note'].startswith('Observation-visible')
    assert not out.provenance['future_trajectory_exposed']
    assert out.agents[0].history[-1][5]==1
    assert out.map_data is not sc.map_data


def test_corridor_reflects_map_and_trajectories_consistently():
    sc=Scene('s2','WOMD_SCENARIO_PROTO',.1,'E',[
        _agent('E',0,0),_agent('A',42,-3.5),_agent('B',4,-3.5),_agent('C',-20,-3.5)],
        {'history_only':True,'future_trajectory_exposed':False},
        {'polylines':[{'kind':'lane','feature_id':23,'xy':[[0.,-3.5],[20.,-3.5]]}]})
    out,_=corridor_subset(sc)
    assert out is not None
    assert out.map_data['polylines'][0]['xy']==[[0.,3.5],[20.,3.5]]
    assert next(a for a in out.agents if a.agent_id=='B').y==3.5


def test_corridor_localizes_tfexample_map_samples_too():
    sc=Scene('s3','WOMD_TF_EXAMPLE',.1,'E',[
        _agent('E',100,100),_agent('A',142,96.5),_agent('B',104,96.5),
        _agent('C',80,96.5)],
        {'history_only':True,'future_trajectory_exposed':False},
        {'samples':[{'xy':[120.,96.5], 'kind':'lane_point'}],
         'observed_signals':[]})
    out,_=corridor_subset(sc)
    assert out is not None
    assert out.map_data['samples'][0]['xy']==[20.,3.5]
    assert out.provenance['map_preserved']
