from caver.models.boundary_gate import DelayEvidence, latency_adjusted_margin, decide_anchor_reuse


def test_latency_can_change_reuse_decision():
    e = DelayEvidence(10., 5., 0., 0.2, q_speed_mps=5., complete_guard_envelope=True,
                      supported_policy_family=True)
    assert latency_adjusted_margin(e) == 9.0
    assert decide_anchor_reuse(e)['decision'] == 'REUSE'
    late = DelayEvidence(10., 5., 0., 2.1, q_speed_mps=5., complete_guard_envelope=True,
                         supported_policy_family=True)
    assert decide_anchor_reuse(late)['decision'] == 'RECOMPUTE'


def test_missing_support_abstains():
    e = DelayEvidence(10., 0., 0., 0.1)
    out = decide_anchor_reuse(e)
    assert out['decision'] == 'ABSTAIN'
    assert out['certificate'] is False

