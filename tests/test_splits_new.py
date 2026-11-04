from caver.data.splits import partition,audit_record_splits
from caver.sim.traffic import standard_fourcar

def record(name,group):
    return {'scene_id':name,'scene':{'provenance':{'overlap_group':group}, 'scene_id':name}}

def test_same_map_group_goes_same_split():
    a=standard_fourcar();b=standard_fourcar()
    for s in (a,b):
        s.provenance['overlap_group']='MAP_LANE_SET:abc'
    a.scene_id='one';b.scene_id='two'
    assert partition(a)==partition(b)

def test_group_overlap_blocks_same_log_windows():
    result=audit_record_splits({'train':[record('a','train')],
                'calibration':[record('b','same')], 'test':[record('c','same')]})
    assert not result['passed']
    assert result['overlap_group_overlap']['calibration-test']==['same']
