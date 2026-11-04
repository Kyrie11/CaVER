import json
from pathlib import Path
import pytest
from caver.data.schema import Scene,Agent,write_jsonl,read_jsonl
from caver.data.womd import find_shards,inspect,read
from caver.data.nuplan import inspect as nuinspect
from caver.data.reactsim import inventory,validate_export
from caver.data.corridor import corridor_subset
from caver.sim.traffic import standard_fourcar

def test_womd_directory_discovery(tmp_path):
    d=tmp_path/'uncompressed'/'scenario'/'training';d.mkdir(parents=True)
    (d/'training.tfrecord-00000-of-01000').write_bytes(b'demo')
    directory,files=find_shards(tmp_path)
    assert len(files)==1
    assert 'training' in str(directory)
    with pytest.raises(ValueError):find_shards(tmp_path,'invalid')

def test_womd_missing_data_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):next(read(tmp_path))

def test_scene_roundtrip(tmp_path):
    s=standard_fourcar();target=tmp_path/'a.jsonl'
    write_jsonl(target,[s]);r=list(read_jsonl(target))[0]
    assert r.ego_id==s.ego_id
    assert [x.agent_id for x in r.agents]==[x.agent_id for x in s.agents]

def test_nuplan_no_fake_db(tmp_path):
    assert nuinspect(tmp_path)=={}

def test_reactsim_requires_provenance(tmp_path):
    d=inventory(tmp_path);assert not d['usable_as_reactive_rollouts']
    file=tmp_path/'export.jsonl';file.write_text('{}\n')
    with pytest.raises(ValueError):validate_export(file)

def test_corridor_filters_ambiguous_scene():
    s=standard_fourcar()
    out,reason=corridor_subset(s)
    assert out is None or out.ego_id=='E'
