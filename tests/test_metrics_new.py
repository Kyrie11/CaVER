from caver.validation.metrics import rq_audit
from caver.experiment import make_scene,paired_scene


def test_rq_audit_separates_event_validity_and_union_labels():
    row=paired_scene(make_scene(90004),source_worlds=1,audit_worlds=1)
    row['experimental_split']='test'
    r=rq_audit([row])
    assert r['scenes']==1
    assert r['cbc_downstream_union_reference_risk_mae'] is not None
    assert r['cbc_pair_reference_risk_mae'] is not None
    assert r['cbc_union_reliability']['label']=='SURROGATE_FINITE_K_REFERENCE_MEANS'
    assert r['simulated_ego_safe_downstream_contrast_UNMATCHED'] in (0,1)
