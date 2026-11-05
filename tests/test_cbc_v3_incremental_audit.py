"""Guard against inflation of graph/event claims on unequal baselines."""
import json
from caver.validation.cbc_gate import summarize


def test_novelty_requires_stronger_equal_budget_control(tmp_path):
    results = {
        'test_scenes': 120,
        'reference_worlds_per_family': 96,
        'reference_families': ['a', 'b'],
        'source_worlds_per_candidate': 4,
        'methods': {
            'source': {'matched_kinematic_proxy_scenes': 90, 'critical_pair_ref_sign_resolved': 40,
                       'downstream_union_reference_risk_mae': 0.3},
            'anchor_copy': {'reference_resolved_sign_error_rate': 1.0},
            'anchor_graph': {'reference_resolved_sign_error_rate': 0.30,
                             'downstream_union_reference_risk_mae': 0.1},
            'anchor_full': {'reference_resolved_sign_error_rate': 0.20},
            'full': {'reference_resolved_sign_error_rate': 0.05},
            'no_events': {'reference_resolved_sign_error_rate': 0.05},
        }
    }
    f = tmp_path / 'report.json'
    f.write_text(json.dumps(results))
    gated = summarize(f)
    # A positive comparison to a trivially weak anchor-copy predictor cannot
    # establish any incremental value of a learned temporal graph.
    assert gated['verdict'] == 'DESCRIPTIVE_SIMULATOR_CONDITIONAL_SIGNAL_ONLY'
    assert gated['novelty_verdict'] == 'INCREMENTAL_GRAPH_AND_EVENT_GAIN_NOT_DEMONSTRATED'
    assert not gated['incremental_ablation']['graph_strictly_better_than_anchor_full_point_estimate']
    assert not gated['incremental_ablation']['full_strictly_better_than_no_events_point_estimate']


def test_missing_critical_reference_forces_abstention(tmp_path):
    f=tmp_path/'empty.json'
    f.write_text(json.dumps({'test_scenes':120,'reference_families':['a','b'],
       'reference_worlds_per_family':96,
       'methods':{'source':{'matched_kinematic_proxy_scenes':15,
                            'critical_pair_ref_sign_resolved':0}}}))
    got=summarize(f)
    assert got['novelty_verdict']=='NOT_TESTABLE_WITH_CURRENT_EVIDENCE'
    assert 'REFERENCE_RESOLVED_CRITICAL_PAIRS_BELOW_DECLARED_MINIMUM' in got['evidence_gaps']
