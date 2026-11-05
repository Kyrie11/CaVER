"""Non-regression gates for the CBC-first research iteration."""
import copy
import gzip
import json
import numpy as np
import pytest
import torch
from caver.data.schema import read_jsonl
from caver.data.splits import training_partition
from caver.data.preparation import prepare_womd
from caver.experiment import paired_scene, stress_scene, stress_actions
from caver.models.anchor import build_anchor_view
from caver.models.temporal import DelayedGraphEncoder
from caver.validation.cbc_study import study as cbc_study
from caver.validation.boundary_study import study as boundary_study


def _training_observations():
    """Return actual pre-intervention Scene objects with stable grouping IDs."""
    from caver.sim.traffic import standard_fourcar
    result=[]
    for i in range(30):
        sc=standard_fourcar()
        sc.scene_id=f'fake-training-{i}'
        sc.provenance.update(history_only=True, future_trajectory_exposed=False,
                             split='training',overlap_group='MAP_LANE_SET:'+str(i),
                             overlap_group_quality='HEURISTIC_MAP_NOT_VERIFIED_PARENT')
        if training_partition(sc)=='train':result.append(sc)
        if len(result)==3:break
    assert len(result)==3
    return result


def test_small_streaming_womd_prep_skips_future_and_keeps_manifest(monkeypatch,tmp_path):
    from caver.data import preparation
    from pathlib import Path
    observed=_training_observations()
    monkeypatch.setattr(preparation,'find_shards',lambda *args:(Path('/fake/training'),[Path('shard.tfrecord')]))
    monkeypatch.setattr(preparation,'read',lambda *args,**kwargs:iter(observed))
    target=tmp_path/'tiny.jsonl.gz'
    info=prepare_womd('/not/accessed','training','train','policy',target,
                      limit=3,max_scan=3,max_output_mb=.02)
    assert info['selected']==3 and info['future_labels_accessed'] is False
    assert target.stat().st_size < .02*1024*1024
    read=list(read_jsonl(target))
    assert len(read)==3
    assert all(sc.map_data is None for sc in read)
    assert all(sc.provenance['preparation_partition']=='train' for sc in read)
    assert all(sc.provenance['future_trajectory_exposed'] is False for sc in read)
    assert target.with_name(target.name+'.manifest.json').is_file()


def test_womd_prep_refuses_invalid_partitions(tmp_path):
    with pytest.raises(ValueError,match='training must'):
        prepare_womd('/empty','training','test','corridor',tmp_path/'bad.jsonl.gz')
    with pytest.raises(ValueError,match='train partition only'):
        prepare_womd('/empty','validation','test','policy',tmp_path/'bad.jsonl.gz')


def test_anchor_view_never_uses_non_anchor_reactor_or_reference_labels():
    sc=stress_scene(10123)
    row=paired_scene(sc, actions=stress_actions(), source_family='stress_source',
                     source_worlds=1,audit_families=('stress_reference',),audit_worlds=1,seed=44)
    anchor0=build_anchor_view(row,'early')
    corrupted=copy.deepcopy(row)
    for key in list(corrupted['diagnostics']):
        if not key.startswith('early/'):
            corrupted['diagnostics'][key]['features']=[-999.]*19
    corrupted['temporal']['late']['nodes']=[[[999.]*6]]
    corrupted['temporal']['reference']['nodes']=[[[999.]*6]]
    corrupted['audited']={k:.999 for k in corrupted['audited']}
    assert build_anchor_view(corrupted,'early')==anchor0
    assert anchor0['source'].startswith('ONE_ANCHOR')


def test_delayed_graph_does_not_propagate_future_into_past():
    torch.manual_seed(7)
    net=DelayedGraphEncoder(channels=12,depth=3).eval()
    a=torch.randn(12,3,6)
    b=a.clone();b[7,0,:]+=300
    edges=[[0,1,2],[1,2,1],[2,1,1]]
    with torch.no_grad():
        first=net(a,edges)
        second=net(b,edges)
    assert torch.allclose(first[:7],second[:7],atol=0,rtol=0)
    assert not torch.allclose(first[9:],second[9:])
    with pytest.raises(ValueError,match='Invalid or noncausal'):
        net(a,[[0,1,0]])


def test_cbc_study_has_small_bounded_output_and_rejects_non_test(tmp_path):
    rows=[]
    from caver.experiment import store_records
    for i in (30001,30002):
        r=paired_scene(stress_scene(i),actions=stress_actions(),source_family='stress_source',
                       source_worlds=1,audit_families=('stress_reference','stress_shifted'),
                       audit_worlds=2,seed=i)
        r['experimental_split']='test'
        rows.append(r)
    p=tmp_path/'test.jsonl.gz';store_records(rows,p)
    result=cbc_study(p,{},bootstrap=8)
    assert result['test_scenes']==2
    assert set(['source','anchor_copy']) <= set(result['methods'])
    assert len(json.dumps(result).encode())<60_000
    rows[0]['experimental_split']='train';store_records(rows,p)
    with pytest.raises(ValueError,match='held-out'):
        cbc_study(p,{})


def test_boundary_study_cannot_reuse_test_scenes_as_development(tmp_path):
    from caver.experiment import store_records
    r=paired_scene(stress_scene(123450),actions=stress_actions(),source_family='stress_source',
                   source_worlds=1,audit_families=('stress_reference',),audit_worlds=1,seed=123)
    r['experimental_split']='train';a=tmp_path/'train.jsonl.gz';store_records([r],a)
    r['experimental_split']='test';b=tmp_path/'test.jsonl.gz';store_records([r],b)
    with pytest.raises(ValueError,match='overlap'):
        boundary_study(a,b)


def test_carla_diagnostic_never_confuses_module_with_rpc(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None)
    from caver.integrations.carla_runner import preflight
    p=preflight()
    assert p['python_module_found'] is False
    assert p['rpc_connected'] is False
    assert 'fourcar-challenge' in p['hint']


def test_no_events_ablation_masks_joint_head_as_well_as_pair_head():
    from caver.models.cbc import CascadeRiskHead,JointRiskHead,EVENT_FEATURES,FEATURES
    torch.manual_seed(71)
    model=CascadeRiskHead(variant='no_events').eval()
    joint=JointRiskHead(suppress_events=True).eval()
    x=torch.randn(4,len(FEATURES));x[:,FEATURES.index('source_pair_contact')]=.23
    x[:,FEATURES.index('source_downstream_union')]=.3
    x[:,FEATURES.index('source_ego_contact')]=.08
    xx=x.clone()
    xx[:,[FEATURES.index(k) for k in EVENT_FEATURES]]+=1000
    with torch.no_grad():
        assert torch.allclose(model(x),model(xx),atol=0,rtol=0)
        assert torch.allclose(joint(x),joint(xx),atol=0,rtol=0)


def test_source_temporal_graph_invariant_to_mc_sample_order():
    from caver.models.temporal import source_temporal_graph
    from caver.sim.traffic import rollout,sample_world
    sc=stress_scene(981)
    a=stress_actions()[0]
    rr=[rollout(sc,a,sample_world(sc,i,'stress_source')) for i in (1,2,3)]
    forward=source_temporal_graph(sc,rr)
    reverse=source_temporal_graph(sc,rr[::-1])
    assert forward['actors']==reverse['actors']
    assert forward['edges']==reverse['edges']
    assert np.allclose(forward['nodes'],reverse['nodes'],atol=2e-7)
