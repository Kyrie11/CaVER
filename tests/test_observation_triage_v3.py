from caver.validation.observation_triage import observation_audit, _reasons
from caver.sim.traffic import standard_fourcar
from caver.data.schema import write_jsonl
import pytest


def test_triage_requires_planning_only_and_has_plausible_outputs(tmp_path):
    sc=standard_fourcar()
    sc.provenance['history_only']=True
    sc.provenance['future_trajectory_exposed']=False
    r=_reasons(sc)
    assert r['bc_gap_m']>0
    assert '0.8' in r['bc_bounded_slack_proxy_m']
    p=tmp_path/'x.jsonl.gz'
    write_jsonl(p,[sc])
    out=observation_audit(p)
    assert out['observed_scenes']==1
    assert out['quantiles']['bc_ttc_s']['n']==1
    sc.provenance['future_trajectory_exposed']=True
    with pytest.raises(ValueError):_reasons(sc)
