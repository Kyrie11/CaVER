from caver.validation.candidate_design import study,_assess
from caver.sim.traffic import standard_fourcar,candidates
from caver.data.schema import write_jsonl
import pytest

def test_action_design_reports_cohort_without_rollouts(tmp_path):
    sc=standard_fourcar()
    sc.provenance.update(history_only=True,future_trajectory_exposed=False)
    p=tmp_path/'case.jsonl.gz';write_jsonl(p,[sc])
    out=study(p)
    assert out['scenes']==1
    assert out['policies']['fixed']['evaluated']==1
    assert out['policies']['observed_adaptive']['evaluated']==1
    assert out['lawful_route_verified'] is False
    assert isinstance(_assess(sc,candidates())['reasons'],list)
    sc.provenance['future_trajectory_exposed']=True
    write_jsonl(p,[sc])
    with pytest.raises(ValueError):study(p)
