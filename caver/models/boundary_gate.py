"""Latency-aware, conservative anchor-reuse gate.

This module is deliberately separate from the learned CBC heads.  It turns
measured planning latency and a declared response envelope into an explicit
reuse decision.  Missing guard coverage defaults to recomputation/abstention;
the function never returns a safety certificate.
"""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class DelayEvidence:
    """Evidence for one directed interaction at the current planning instant."""

    clearance_m: float
    closing_speed_mps: float
    closing_accel_mps2: float
    total_latency_s: float
    q_speed_mps: float | None = None
    q_accel_mps2: float | None = None
    guard_error_m: float = 0.0
    complete_guard_envelope: bool = False
    supported_policy_family: bool = False


def latency_adjusted_margin(evidence: DelayEvidence) -> float:
    """Conservative occupancy margin after the full measured callback delay."""
    values = asdict(evidence)
    if any(not math.isfinite(float(values[k])) for k in
           ('clearance_m', 'closing_speed_mps', 'closing_accel_mps2', 'total_latency_s', 'guard_error_m')):
        raise ValueError('non-finite delay evidence')
    if evidence.total_latency_s < 0. or evidence.guard_error_m < 0.:
        raise ValueError('latency and guard error must be non-negative')
    speed = max(0., float(evidence.q_speed_mps if evidence.q_speed_mps is not None
                           else evidence.closing_speed_mps))
    accel = max(0., float(evidence.q_accel_mps2 if evidence.q_accel_mps2 is not None
                           else evidence.closing_accel_mps2))
    t = float(evidence.total_latency_s)
    return float(evidence.clearance_m - speed * t - .5 * accel * t * t - evidence.guard_error_m)


def decide_anchor_reuse(evidence: DelayEvidence, tolerance_m: float = 0.0) -> dict:
    """Return ``REUSE`` only outside the declared latency/guard boundary.

    ``RECOMPUTE`` is an operational recommendation, not a guarantee that the
    recomputed candidate is safe.  ``ABSTAIN`` is used when the envelope or
    policy support is missing.
    """
    if tolerance_m < 0. or not math.isfinite(float(tolerance_m)):
        raise ValueError('tolerance_m must be finite and non-negative')
    if not evidence.complete_guard_envelope or not evidence.supported_policy_family:
        return {'decision': 'ABSTAIN', 'reason': 'MISSING_GUARD_OR_POLICY_SUPPORT',
                'margin_m': None, 'latency_s': evidence.total_latency_s,
                'certificate': False}
    margin = latency_adjusted_margin(evidence)
    if margin <= tolerance_m:
        return {'decision': 'RECOMPUTE', 'reason': 'LATENCY_ADJUSTED_MARGIN_BOUNDARY',
                'margin_m': margin, 'latency_s': evidence.total_latency_s,
                'certificate': False}
    return {'decision': 'REUSE', 'reason': 'OUTSIDE_DECLARED_LATENCY_GUARD_BOUNDARY',
            'margin_m': margin, 'latency_s': evidence.total_latency_s,
            'certificate': False}

