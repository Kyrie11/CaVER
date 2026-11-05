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
    behavior_delay_accounted_s: float = 0.0
    callback_latency_s: float | None = None
    transport_latency_s: float | None = None
    graph_latency_s: float | None = None
    fallback_latency_s: float | None = None
    arrival_cdf_lower_bound: float | None = None
    arrival_confidence_delta: float = .05
    latency_measured: bool = True


def latency_adjusted_margin(evidence: DelayEvidence) -> float:
    """Conservative occupancy margin after the full measured callback delay."""
    values = asdict(evidence)
    if any(not math.isfinite(float(values[k])) for k in
           ('clearance_m', 'closing_speed_mps', 'closing_accel_mps2', 'total_latency_s', 'guard_error_m',
            'behavior_delay_accounted_s')):
        raise ValueError('non-finite delay evidence')
    if evidence.total_latency_s < 0. or evidence.guard_error_m < 0. or evidence.behavior_delay_accounted_s < 0.:
        raise ValueError('latency and guard error must be non-negative')
    for optional in ('q_speed_mps', 'q_accel_mps2'):
        value=getattr(evidence, optional)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f'{optional} must be finite when supplied')
    speed = max(0., float(evidence.q_speed_mps if evidence.q_speed_mps is not None
                           else evidence.closing_speed_mps))
    accel = max(0., float(evidence.q_accel_mps2 if evidence.q_accel_mps2 is not None
                           else evidence.closing_accel_mps2))
    # The reactive source rollout already accounts for driver behavior delay;
    # only residual evidence age is charged to the reuse guard.
    t = max(0., float(evidence.total_latency_s) - float(evidence.behavior_delay_accounted_s))
    return float(evidence.clearance_m - speed * t - .5 * accel * t * t - evidence.guard_error_m)


def decide_anchor_reuse(evidence: DelayEvidence, tolerance_m: float = 0.0) -> dict:
    """Return ``REUSE`` only outside the declared latency/guard boundary.

    ``RECOMPUTE`` is an operational recommendation, not a guarantee that the
    recomputed candidate is safe.  ``ABSTAIN`` is used when the envelope or
    policy support is missing.
    """
    if tolerance_m < 0. or not math.isfinite(float(tolerance_m)):
        raise ValueError('tolerance_m must be finite and non-negative')
    if not evidence.latency_measured:
        return {'decision': 'ABSTAIN', 'reason': 'UNMEASURED_CALLBACK_LATENCY',
                'margin_m': None, 'latency_s': evidence.total_latency_s,
                'residual_latency_s': None, 'certificate': False}
    if not evidence.complete_guard_envelope or not evidence.supported_policy_family:
        return {'decision': 'ABSTAIN', 'reason': 'MISSING_GUARD_OR_POLICY_SUPPORT',
                'margin_m': None, 'latency_s': evidence.total_latency_s,
                'residual_latency_s': None,
                'certificate': False}
    if evidence.arrival_cdf_lower_bound is not None:
        if not (0. <= evidence.arrival_cdf_lower_bound <= 1. and
                0. < evidence.arrival_confidence_delta < 1.):
            raise ValueError('arrival CDF lower bound or confidence delta is invalid')
        if evidence.arrival_cdf_lower_bound < 1. - evidence.arrival_confidence_delta:
            return {'decision': 'RECOMPUTE', 'reason': 'RESPONSE_ARRIVAL_CDF_INCOMPLETE',
                    'margin_m': None, 'latency_s': evidence.total_latency_s,
                    'residual_latency_s': max(0., evidence.total_latency_s - evidence.behavior_delay_accounted_s),
                    'arrival_cdf_lower_bound': evidence.arrival_cdf_lower_bound,
                    'certificate': False}
    try:
        margin = latency_adjusted_margin(evidence)
    except ValueError as exc:
        return {'decision': 'ABSTAIN', 'reason': 'INVALID_DELAY_EVIDENCE',
                'margin_m': None, 'latency_s': evidence.total_latency_s,
                'residual_latency_s': None, 'error': str(exc), 'certificate': False}
    residual=max(0., evidence.total_latency_s - evidence.behavior_delay_accounted_s)
    components={k:getattr(evidence,k) for k in ('callback_latency_s','transport_latency_s',
                                                'graph_latency_s','fallback_latency_s')
                 if getattr(evidence,k) is not None}
    if margin <= tolerance_m:
        return {'decision': 'RECOMPUTE', 'reason': 'LATENCY_ADJUSTED_MARGIN_BOUNDARY',
                'margin_m': margin, 'latency_s': evidence.total_latency_s,
                'residual_latency_s': residual, 'latency_components_s': components,
                'certificate': False}
    return {'decision': 'REUSE', 'reason': 'OUTSIDE_DECLARED_LATENCY_GUARD_BOUNDARY',
            'margin_m': margin, 'latency_s': evidence.total_latency_s,
            'residual_latency_s': residual, 'latency_components_s': components,
            'certificate': False}

