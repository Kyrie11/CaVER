from cbc_extended_audit import (
    critical_pair_audit, reference_flip_audit,
    matched_legal_ego_safe_yield, false_safe_reuse_audit,
    event_detection_audit, rare_risk_calibration, cost_rollup,
)


def _row():
    return {
        "scene_id": "s0",
        "audit_families": ["f0"],
        "audit_worlds_per_family": 100,
        "actions": [{"action_id": "early"}, {"action_id": "late"}],
        "pairs": ["A|B"],
        "predicted": {"early/A|B": .20, "late/A|B": .10,
                      "early/down": .10, "late/down": .20,
                      "early/ego": 0., "late/ego": 0.},
        "audited": {"early/A|B": .80, "late/A|B": .20,
                     "early/down": .80, "late/down": .20,
                     "early/ego": 0., "late/ego": 0.},
        "candidate_matching": {"pairs": {"early|late": {
            "matched_kinematic_proxy": True,
            "lawful_route_match": True,
        }}},
        "boundary_diagnostics": {"A|B": {
            "reference": {"contact_flip_rate": .20,
                           "n_paired_latent_worlds": 100},
        }},
        "reuse_decisions": [{"accepted": True, "predicted_risk": .02,
                              "reference_risk": .80, "stratum": "near"}],
        "events": [{"reference_present": True, "predicted_present": True,
                    "reference_onset_s": 2., "predicted_onset_s": 1.5,
                    "stratum": "near"}],
        "timing": {"total_ms": 10., "cbc_ms": 1.},
        "callback_count": 1,
        "rollout_count": 8,
    }


def test_sign_and_underresolution_are_separate():
    row = _row()
    out = critical_pair_audit([row], contrast_threshold=.2, detect_threshold=.05)
    assert out["critical_pair_tests"] == 1
    assert out["sign_error_rate_all_critical"] == 0.0
    assert out["underresolution_rate"] == 0.0
    # Reversing the prediction should be a sign error, not a miss.
    row["predicted"]["early/A|B"] = .10
    row["predicted"]["late/A|B"] = .20
    out = critical_pair_audit([row], contrast_threshold=.2, detect_threshold=.05)
    assert out["sign_error_rate_all_critical"] == 1.0
    assert out["underresolution_rate"] == 0.0


def test_flip_ci_yield_false_safe_events_calibration_and_cost():
    row = _row()
    flip = reference_flip_audit([row], n_boot=100)
    assert flip["flip_count"] == 1
    assert flip["pair_mc_ci_available"] == 1
    assert flip["scene_cluster_bootstrap"]["point"] == 1.0
    y = matched_legal_ego_safe_yield([row])
    assert y["matched_yield"] == 1.0
    assert y["legal_given_matched_yield"] == 1.0
    assert y["ego_safe_given_matched_yield"] == 1.0
    fs = false_safe_reuse_audit([row])
    assert fs["false_safe_rate_among_reused"] == 1.0
    ev = event_detection_audit([row])
    assert ev["recall"] == 1.0
    assert ev["lead_time_s"]["p50"] == .5
    cal = rare_risk_calibration([row])
    assert cal["status"] == "MEASURED"
    cost = cost_rollup([row])
    assert cost["status"] == "MEASURED"
    assert cost["total_wall_time_ms"]["p95"] == 10.0


def test_missing_evidence_is_not_invented():
    row = {"scene_id": "no-evidence", "actions": [], "pairs": []}
    assert reference_flip_audit([row])["status"] == "NOT_MEASURED"
    assert false_safe_reuse_audit([row])["status"] == "NOT_MEASURED"
    assert event_detection_audit([row])["status"] == "NOT_MEASURED"
    assert cost_rollup([row])["status"] == "NOT_MEASURED"
    y = matched_legal_ego_safe_yield([row])
    assert y["matched_yield"] == 0.0
    assert y["status"] == "NOT_MEASURED"
