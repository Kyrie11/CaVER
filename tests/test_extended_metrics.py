from caver.experiment import make_scene, paired_scene
from caver.validation.extended_metrics import extended_audit, critical_pair_metrics, yield_summary


def test_extended_audit_keeps_unresolved_and_sign_error_separate():
    row = paired_scene(make_scene(91001), source_worlds=2, audit_worlds=2)
    row['experimental_split'] = 'test'
    out = extended_audit([row], bootstrap=30)
    assert out['schema'] == 'CBC_EXTENDED_AUDIT_V1'
    cp = out['critical_pairs']
    assert 'critical_pair_sign_error_rate' in cp
    assert 'critical_pair_underresolution_rate' in cp
    assert out['event_timing']['status'] == 'NOT_MEASURED'
    assert out['false_safe_anchor_reuse']['status'] == 'NOT_MEASURED'
    assert out['runtime']['status'] == 'MEASURED'
    assert out['runtime']['scope'].startswith('OFFLINE_')


def test_yield_does_not_infer_route_legality():
    row = paired_scene(make_scene(91002), source_worlds=1, audit_worlds=1)
    row['experimental_split'] = 'test'
    y = yield_summary([row])
    assert y['legal_route']['status'] == 'NOT_MEASURED'
    assert y['matched_legal_ego_safe']['status'] == 'NOT_MEASURED'


def test_scene_cluster_bootstrap_is_available_for_reference_flips():
    rows = []
    for seed in (91003, 91004, 91005):
        row = paired_scene(make_scene(seed), source_worlds=1, audit_worlds=2)
        row['experimental_split'] = 'test'
        rows.append(row)
    out = critical_pair_metrics(rows, bootstrap=20)
    ci = out['reference_pair_flip']['scene_cluster_bootstrap_flip_rate']
    assert ci['status'] == 'MEASURED'
    assert ci['unit'] == 'scene_cluster'

