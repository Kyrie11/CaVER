"""Extended CBC diagnostics with explicit denominators and uncertainty.

The original pilot reports are useful model-conditional diagnostics, but they
do not separate a wrong sign from an unresolved contrast and do not expose
the denominator needed for a false-safe reuse claim.  This module intentionally
keeps those quantities separate.  It also refuses to manufacture legal-route,
event-timing, or wall-clock claims when the record schema does not contain the
required evidence.

All reference targets in this module are finite-Monte-Carlo means from the
declared simulator/reference law.  They are not real-world counterfactual
labels, and the bootstrap only clusters dependence at the scene level.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Callable, Iterable

import numpy as np


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _quantile(values, q):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(values, q)) if values else None


def _sign(x: float, tol: float = 0.0) -> int:
    return 1 if x > tol else -1 if x < -tol else 0


def wilson_interval(successes: int, trials: int, confidence: float = .95):
    """Wilson interval for a binomial proportion, including zero trials."""
    if trials <= 0:
        return None
    # Normal quantiles are enough for the declared descriptive interval.  The
    # exact Clopper--Pearson interval is used when scipy is available below.
    z = 1.959963984540054 if confidence == .95 else 1.96
    p = successes / trials
    den = 1. + z * z / trials
    ctr = (p + z * z / (2. * trials)) / den
    rad = z * math.sqrt(p * (1. - p) / trials + z * z / (4. * trials * trials)) / den
    return [max(0., ctr - rad), min(1., ctr + rad)]


def clopper_pearson(successes: int, trials: int, confidence: float = .95):
    """Exact binomial interval, with a dependency-light Wilson fallback."""
    if trials <= 0:
        return None
    try:
        from scipy.stats import beta
        alpha = 1. - confidence
        lo = 0. if successes == 0 else float(beta.ppf(alpha / 2., successes, trials - successes + 1))
        hi = 1. if successes == trials else float(beta.ppf(1. - alpha / 2., successes + 1, trials - successes))
        return [lo, hi]
    except Exception:  # pragma: no cover - scipy is a project dependency
        return wilson_interval(successes, trials, confidence)


def _cluster_bootstrap(scene_values: dict[str, float], statistic: Callable[[list[float]], float],
                       n_boot: int = 2000, seed: int = 51):
    """Scene-cluster bootstrap interval; pair rows within a scene stay together."""
    vals = [float(v) for v in scene_values.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2 or n_boot <= 0:
        return {'status': 'NOT_MEASURED', 'n_scenes': len(vals), 'ci95': None}
    rng = np.random.default_rng(seed)
    samples = []
    n = len(vals)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        samples.append(float(statistic([vals[i] for i in idx])))
    return {'status': 'MEASURED', 'n_scenes': n,
            'estimate': float(statistic(vals)),
            'ci95': [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            'bootstrap_resamples': int(n_boot), 'unit': 'scene_cluster'}


def _actions(row):
    return [a['action_id'] for a in row.get('actions', [])]


def _matching(row, a='early', b='late'):
    pairs = row.get('candidate_matching', {}).get('pairs', {})
    return pairs.get('|'.join(sorted((a, b))), {})


def _ego_safe(row, a='early', b='late', threshold=.05):
    aud = row.get('audited', {})
    try:
        return float(aud[f'{a}/ego']) <= threshold and float(aud[f'{b}/ego']) <= threshold
    except (KeyError, TypeError, ValueError):
        return None


def yield_summary(rows: Iterable[dict[str, Any]], ego_threshold: float = .05):
    """Report pre-label candidate yields without guessing route legality."""
    rows = list(rows)
    n = len(rows)
    matched = []
    intent = []
    dynamics = []
    legal = []
    ego_safe = []
    for row in rows:
        m = _matching(row)
        intent.append(bool(m.get('same_maneuver_intent')) if 'same_maneuver_intent' in m else None)
        dynamics.append(bool(m.get('kinematic_feasible_both')) if 'kinematic_feasible_both' in m else None)
        matched.append(bool(m.get('matched_kinematic_proxy')) if 'matched_kinematic_proxy' in m else None)
        # The current schema explicitly stores NOT_EVALUATED.  Unknown is kept
        # unknown; treating it as legal would inflate the paper's denominator.
        law = m.get('lawful_route_match', row.get('candidate_selection_audit', {}).get('lawful_route_feasibility'))
        legal.append(law if isinstance(law, bool) else None)
        ego_safe.append(_ego_safe(row, threshold=ego_threshold))

    def count_known(values):
        known = [v for v in values if v is not None]
        return {'count': int(sum(known)), 'denominator': len(known),
                'yield': (float(sum(known) / len(known)) if known else None),
                'status': 'MEASURED' if known else 'NOT_MEASURED'}

    joint = [bool(m and l and e) for m, l, e in zip(matched, legal, ego_safe)
             if m is not None and l is not None and e is not None]
    return {
        'candidate_pairs': n,
        'intent_matched': count_known(intent),
        'kinematic_feasible_both': count_known(dynamics),
        'matched_kinematic_proxy': count_known(matched),
        'legal_route': count_known(legal),
        'both_ego_safe_under_reference': count_known(ego_safe),
        'matched_legal_ego_safe': {
            'count': int(sum(joint)), 'denominator': len(joint),
            'yield': float(sum(joint) / len(joint)) if joint else None,
            'status': 'MEASURED' if joint else 'NOT_MEASURED',
            'reason': 'legal_route_match is absent or NOT_EVALUATED in current records'
            if not joint else None,
        },
        'warning': 'Nominal geometric clearance is not reference ego safety; route legality is never inferred.'
    }


FINITE_MC_RADIUS_SCHEMA = "paired_bernoulli_hoeffding_fwer_v1"
FINITE_MC_RADIUS_FORMULA = "sqrt(2*log(2*n_pairs/alpha)/n_worlds)"

def finite_mc_radius(n_worlds: int, n_pairs: int, alpha: float = .05) -> float:
    """Conservative within-scene finite-MC radius for paired Bernoulli deltas.

    The radius is a resolution threshold, not a confidence interval over scenes.
    The Bonferroni factor covers the declared pair comparisons in one scene.
    """
    n_worlds = int(n_worlds); n_pairs = max(1, int(n_pairs))
    if n_worlds <= 0:
        return math.inf
    if not 0. < alpha < 1.:
        raise ValueError("alpha must lie in (0,1)")
    return math.sqrt(2. * math.log(2. * n_pairs / alpha) / n_worlds)

def finite_mc_radius_metadata(n_worlds: int, n_pairs: int, alpha: float = .05):
    return {"schema": FINITE_MC_RADIUS_SCHEMA, "formula": FINITE_MC_RADIUS_FORMULA,
            "alpha": float(alpha), "n_worlds": int(n_worlds),
            "n_pairs": int(n_pairs), "radius": finite_mc_radius(n_worlds, n_pairs, alpha)}


def _mc_radius(row, familywise=True):
    n = int(row.get('audit_worlds_per_family', 0)) * len(row.get('audit_families', []))
    pairs = max(1, len(row.get('pairs', [])))
    # The non-familywise option is retained for API compatibility but uses the
    # same declared radius; all current critical decisions are familywise.
    return finite_mc_radius(n, pairs, .05)


def critical_pair_metrics(rows: Iterable[dict[str, Any]], prediction_key: str = 'predicted',
                          contrast_threshold: float = .20, detection_threshold: float = .05,
                          ego_threshold: float = .05, bootstrap: int = 2000, seed: int = 51):
    """Separate sign error, underresolution and critical miss.

    The reference-resolved denominator is fixed before inspecting predicted
    values.  A direction error requires a nonzero, resolved prediction; a
    prediction inside the detection deadband is reported as underresolution.
    """
    rows = list(rows)
    events = []
    all_critical = []
    by_scene = defaultdict(list)
    for row in rows:
        acts = _actions(row)
        if not {'early', 'late'} <= set(acts):
            continue
        match = _matching(row)
        matched = bool(match.get('matched_kinematic_proxy', False))
        ego_safe = _ego_safe(row, threshold=ego_threshold)
        # Legal evidence is optional in the historical schema.  If it exists,
        # legal must be true; if absent, retain the row for a simulator-only
        # diagnostic but mark the legal denominator as unknown elsewhere.
        law = match.get('lawful_route_match')
        legal_ok = law is True or law in (None, 'NOT_EVALUATED', 'UNVERIFIED')
        if not matched or ego_safe is not True or not legal_ok:
            continue
        pred = row.get(prediction_key, {})
        aud = row.get('audited', {})
        for pair in row.get('pairs', []):
            try:
                truth = float(aud[f'early/{pair}']) - float(aud[f'late/{pair}'])
                estimate = float(pred[f'early/{pair}']) - float(pred[f'late/{pair}'])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(truth) < contrast_threshold:
                continue
            radius = _mc_radius(row, familywise=True)
            resolved = abs(truth) > radius
            under = abs(estimate) < detection_threshold
            sign_error = (not under) and _sign(truth) != _sign(estimate)
            miss = under or sign_error
            item = {'scene_id': row.get('scene_id'), 'pair': pair,
                    'true_delta': truth, 'pred_delta': estimate,
                    'reference_mc_radius': radius,
                    'reference_resolved': bool(resolved),
                    'underresolution': bool(under),
                    'sign_error': bool(sign_error),
                    'critical_miss': bool(miss),
                    'matched': matched, 'legal': law if isinstance(law, bool) else None,
                    'ego_safe': True}
            all_critical.append(item)
            if resolved:
                events.append(item)
                by_scene[str(row.get('scene_id'))].append(item)

    def rate(key, xs):
        return float(np.mean([float(x[key]) for x in xs])) if xs else None

    scene_rates = {}
    for sid, xs in by_scene.items():
        scene_rates[sid] = {
            k: rate(k, xs) for k in ('sign_error', 'underresolution', 'critical_miss')
        }
    cluster = {}
    for key in ('sign_error', 'underresolution', 'critical_miss'):
        cluster[key] = _cluster_bootstrap(
            {sid: vals[key] for sid, vals in scene_rates.items()}, np.mean, bootstrap, seed)
    flips = reference_pair_flip_summary(rows, bootstrap=bootstrap, seed=seed)
    return {
        'critical_pair_tests': len(events),
        'critical_pair_all_tests': len(all_critical),
        'critical_pair_reference_unresolved': int(sum(not x['reference_resolved'] for x in all_critical)),
        'critical_pair_sign_error_rate_all_critical': rate('sign_error', all_critical),
        'critical_pair_underresolution_rate_all_critical': rate('underresolution', all_critical),
        'critical_pair_miss_rate_all_critical': rate('critical_miss', all_critical),
        'critical_pair_sign_error_rate': rate('sign_error', events),
        'critical_pair_underresolution_rate': rate('underresolution', events),
        'critical_pair_miss_rate': rate('critical_miss', events),
        'scene_macro_rates': {
            k: _mean([v[k] for v in scene_rates.values()]) if scene_rates else None
            for k in ('sign_error', 'underresolution', 'critical_miss')
        },
        'scene_cluster_bootstrap': cluster,
        'events': events,
        'all_critical_events': all_critical,
        'reference_pair_flip': flips,
        'definition': {
            'critical': f'|delta_ref| >= {contrast_threshold}',
            'reference_resolved': 'critical contrast exceeds conservative finite-MC Hoeffding radius',
            'sign_error': 'resolved and prediction is outside detection deadband with opposite sign',
            'underresolution': f'resolved and |delta_pred| < {detection_threshold}',
            'critical_miss': 'sign_error OR underresolution',
            'uncertainty_unit': 'scene cluster; pair rows inside a scene are dependent',
        },
        'warning': 'Legal route is not inferred when current records mark it NOT_EVALUATED.'
    }


def reference_pair_flip_summary(rows: Iterable[dict[str, Any]], flip_threshold: float = 0.,
                                bootstrap: int = 2000, seed: int = 51):
    """Reference paired-world flip point estimates, MC intervals, and cluster CI."""
    records = []
    by_scene = defaultdict(list)
    for row in rows:
        for pair, diag in row.get('boundary_diagnostics', {}).items():
            ref = diag.get('reference', {})
            rate = ref.get('contact_flip_rate')
            n = int(ref.get('n_paired_latent_worlds', 0))
            if rate is None or n <= 0:
                continue
            rate = float(rate)
            successes = int(round(rate * n))
            item = {'scene_id': row.get('scene_id'), 'pair': pair,
                    'flip_rate': rate, 'n_worlds': n,
                    'flip_ci95_mc': clopper_pearson(successes, n),
                    'flip': bool(rate > flip_threshold)}
            records.append(item)
            by_scene[str(row.get('scene_id'))].append(rate)
    scene_mean = {sid: float(np.mean(vals)) for sid, vals in by_scene.items() if vals}
    scene_flip = {sid: float(np.mean([v > flip_threshold for v in vals]))
                  for sid, vals in by_scene.items() if vals}
    return {
        'pair_comparisons': len(records),
        'flip_pairs': int(sum(x['flip'] for x in records)),
        'flip_fraction_pair_weighted': _mean([float(x['flip']) for x in records]),
        'mean_flip_rate_pair_weighted': _mean([x['flip_rate'] for x in records]),
        'mc_intervals': records,
        'scene_cluster_bootstrap_flip_rate': _cluster_bootstrap(scene_mean, np.mean, bootstrap, seed),
        'scene_cluster_bootstrap_flip_fraction': _cluster_bootstrap(scene_flip, np.mean, bootstrap, seed + 1),
        'scope': 'paired latent worlds from declared reference simulator; not human individual-world causality',
    }


def false_safe_reuse_summary(rows: Iterable[dict[str, Any]], clearance_multiplier: float | None = None,
                             guard_multiplier: float | None = None, flip_threshold: float = 0.,
                             bootstrap: int = 2000, seed: int = 51):
    """Audit empirical reuse only when the frozen screen thresholds are supplied."""
    rows = list(rows)
    # Prefer explicit online decisions when available.  This is the only
    # denominator that can support a system-level false-safe claim.
    explicit = []
    for row in rows:
        decisions = row.get('reuse_decisions')
        if isinstance(decisions, dict):
            decisions = list(decisions.values())
        if not isinstance(decisions, list):
            continue
        for d in decisions:
            if not isinstance(d, dict):
                continue
            accepted = d.get('accepted', d.get('reused'))
            pred = d.get('predicted_risk', d.get('predicted'))
            ref = d.get('reference_risk', d.get('reference'))
            if accepted is None or pred is None or ref is None:
                continue
            explicit.append({'scene_id': row.get('scene_id'), 'accepted': bool(accepted),
                             'predicted': float(pred), 'reference': float(ref),
                             'false_safe': bool(accepted and float(pred) < .05 and float(ref) >= .20),
                             'stratum': str(d.get('stratum', 'all'))})
    if explicit:
        accepted = [x for x in explicit if x['accepted']]
        fs = [x for x in accepted if x['false_safe']]
        return {'status': 'MEASURED', 'source': 'EXPLICIT_REUSE_DECISIONS',
                'decisions': len(explicit), 'accepted': len(accepted), 'false_safe_pairs': len(fs),
                'reuse_fraction': len(accepted) / len(explicit),
                'false_safe_rate_among_reused': len(fs) / len(accepted) if accepted else None,
                'false_safe_rate_all_decisions': len(fs) / len(explicit),
                'strata': {s: {'decisions': sum(x['stratum'] == s for x in explicit),
                               'accepted': sum(x['stratum'] == s and x['accepted'] for x in explicit),
                               'false_safe': sum(x['stratum'] == s and x['false_safe'] for x in explicit)}
                           for s in sorted({x['stratum'] for x in explicit})},
                'definition': 'accepted && predicted_risk < 0.05 && reference_risk >= 0.20'}
    if clearance_multiplier is None or guard_multiplier is None:
        return {'status': 'NOT_MEASURED', 'reason': 'frozen clearance/guard screen thresholds are absent from records'}
    by_scene = defaultdict(lambda: [0, 0])
    items = []
    for row in rows:
        match = _matching(row)
        d_e = float(match.get('d_e', math.inf))
        for pair, diag in row.get('boundary_diagnostics', {}).items():
            ref = diag.get('reference', {})
            flip = float(ref.get('contact_flip_rate', 0.)) > flip_threshold
            margin = float(diag.get('anchor_min_abs_clearance_m', math.inf))
            guard = float(diag.get('anchor_min_abs_brake_guard_mps2', math.inf))
            reuse = margin > clearance_multiplier * d_e and guard > guard_multiplier * d_e
            false_safe = bool(reuse and flip)
            sid = str(row.get('scene_id'))
            by_scene[sid][0] += int(reuse)
            by_scene[sid][1] += int(false_safe)
            items.append({'scene_id': row.get('scene_id'), 'pair': pair,
                          'reuse': bool(reuse), 'reference_flip': bool(flip),
                          'false_safe': false_safe})
    reuse_n = sum(int(x['reuse']) for x in items)
    false_n = sum(int(x['false_safe']) for x in items)
    scene_rates = {sid: (b / a if a else 0.) for sid, (a, b) in by_scene.items()}
    return {
        'status': 'MEASURED', 'pair_comparisons': len(items),
        'reused_pairs': reuse_n, 'false_safe_pairs': false_n,
        'reuse_fraction': reuse_n / len(items) if items else None,
        'false_safe_rate_among_reused': false_n / reuse_n if reuse_n else None,
        'false_safe_scene_cluster_ci95': _cluster_bootstrap(scene_rates, np.mean, bootstrap, seed),
        'thresholds': {'clearance_multiplier': clearance_multiplier, 'guard_multiplier': guard_multiplier},
        'definition': 'reused by frozen screen AND held-out reference paired contact flip',
        'items': items,
    }


def event_timing_summary(rows: Iterable[dict[str, Any]]):
    """Compute event timing with an explicit online/offline scope."""
    rows = list(rows)
    labels = []
    scope = 'ONLINE_TRIGGER_DIAGNOSTIC'
    for row in rows:
        explicit=row.get('event_timing_labels', row.get('events', []))
        if explicit:
            labels.extend(explicit)
            continue
        # Current collectors expose source/reference response onset only.
        # This is useful for reactor diagnostics but is not an online CBC
        # trigger and must not be reported as event-head recall.
        for x in row.get('offline_response_onset_diagnostics', []):
            labels.append({
                'event_type':x.get('event_type','brake_onset'),
                'reference_present':x.get('reference_response_present',False),
                'predicted_present':x.get('source_response_present',False),
                'reference_onset_s':x.get('reference_onset_s'),
                'predicted_trigger_s':x.get('source_response_onset_s'),
                'diagnostic_scope':'OFFLINE_SOURCE_REFERENCE_RESPONSE_ONSET'})
            scope='OFFLINE_SOURCE_REFERENCE_RESPONSE_ONSET'
    if not labels:
        return {'status': 'NOT_MEASURED',
                'reason': 'records contain guard-flip summaries but no frozen event labels and predicted trigger times',
                'required_fields': ['event_type', 'reference_onset_s', 'predicted_trigger_s', 'event_observed']}
    ref_present = [bool(x.get('event_observed', x.get('reference_present',
                        x.get('reference_onset_s', x.get('ref_onset_s')) is not None))) for x in labels]
    pred_present = [bool(x.get('predicted_present',
                         x.get('predicted_trigger_s', x.get('predicted_onset_s')) is not None)) for x in labels]
    tp = sum(a and b for a, b in zip(ref_present, pred_present))
    fp = sum((not a) and b for a, b in zip(ref_present, pred_present))
    fn = sum(a and (not b) for a, b in zip(ref_present, pred_present))
    tn = sum((not a) and (not b) for a, b in zip(ref_present, pred_present))
    observed = [x for x, present in zip(labels, ref_present) if present]
    hits = [x for x, present in zip(labels, pred_present)
            if present and x.get('predicted_trigger_s', x.get('predicted_onset_s')) is not None
            and x.get('reference_onset_s', x.get('ref_onset_s')) is not None]
    lead = [float(x.get('reference_onset_s', x.get('ref_onset_s'))) -
            float(x.get('predicted_trigger_s', x.get('predicted_onset_s'))) for x in hits]
    return {'status': 'MEASURED' if scope=='ONLINE_TRIGGER_DIAGNOSTIC' else 'MEASURED_OFFLINE_RESPONSE_DIAGNOSTIC',
            'scope':scope, 'event_head_recall_measured':scope=='ONLINE_TRIGGER_DIAGNOSTIC',
            'event_count': len(labels), 'observed_count': len(observed),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'detected_count': len(hits), 'recall': tp / (tp + fn) if tp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'miss_rate': fn / (tp + fn) if tp + fn else None,
            'false_alarm_rate': fp / (fp + tn) if fp + tn else None,
            'lead_time_mean_s': _mean(lead), 'lead_time_p50_s': _quantile(lead, .5),
            'lead_time_p95_s': _quantile(lead, .95),
            'convention': 'positive lead means trigger before reference onset'}


def stratified_reference_calibration(rows: Iterable[dict[str, Any]], prediction_key: str = 'predicted', min_count: int = 20):
    """Reliability of finite reference means, stratified by boundary and risk."""
    bins = [0., .01, .05, .2, .5, 1.0000001]
    groups = defaultdict(list)
    for row in rows:
        pred = row.get(prediction_key, {})
        aud = row.get('audited', {})
        # ``reference`` is an attribution baseline rather than a planner
        # candidate and is excluded from CBC candidate calibration.
        for action in (a for a in _actions(row) if a != 'reference'):
            for pair in row.get('pairs', []):
                try:
                    p = float(pred[f'{action}/{pair}']); y = float(aud[f'{action}/{pair}'])
                except (KeyError, TypeError, ValueError):
                    continue
                idx = next((i for i in range(len(bins) - 1) if bins[i] <= p < bins[i + 1]), len(bins) - 2)
                diag = row.get('boundary_diagnostics', {}).get(pair, {})
                near = diag.get('anchor_min_abs_clearance_m') is not None and float(diag['anchor_min_abs_clearance_m']) <= 2.
                groups[(f'risk_bin_{idx}', 'near_boundary' if near else 'far_boundary')].append((p, y))
    out = {}
    for (name, boundary), vals in groups.items():
        ps, ys = zip(*vals)
        out[f'{name}/{boundary}'] = {'count': len(vals), 'mean_prediction': float(np.mean(ps)),
                                      'reference_mean': float(np.mean(ys)),
                                      'mae': float(np.mean(np.abs(np.asarray(ps) - np.asarray(ys)))),
                                      'mse': float(np.mean((np.asarray(ps) - np.asarray(ys)) ** 2)),
                                      'residual_quantiles95': (lambda z: [float(np.quantile(z, .025)), float(np.quantile(z, .975))])(np.asarray(ps)-np.asarray(ys)) if len(vals)>=min_count else None,
                                      'mae_ci95': None,
                                      'count_minimum': int(min_count),
                                      'meets_count_minimum': bool(len(vals)>=min_count),
                                      'label': 'finite_reference_mean_not_binary_ece',
                                      'interpretation': 'descriptive finite-reference mean error; do not call ECE'}
    return {'status': 'MEASURED' if out else 'NOT_MEASURED', 'strata': out,
            'bin_count_minimum': int(min_count),
            'metric_name': 'stratified finite-reference mean error',
            'warning': 'These are descriptive stratified errors for finite reference means, not ECE; bins below the count minimum have no CI.'}


def runtime_summary(rows: Iterable[dict[str, Any]]):
    """Summarize measured record-generation timing, or explicitly abstain."""
    rows = list(rows)
    runtime = [r.get('runtime') for r in rows if isinstance(r.get('runtime'), dict)]
    if not runtime:
        return {'status': 'NOT_MEASURED',
                'reason': 'no monotonic runtime transcript in records',
                'required_fields': ['total_ms', 'source_rollout_calls', 'callback_count', 'reused', 'fallback']}
    def stats(key):
        vals = [float(x[key]) for x in runtime if x.get(key) is not None and np.isfinite(float(x[key]))]
        return {'count': len(vals), 'p50_ms': _quantile(vals, .5), 'p95_ms': _quantile(vals, .95),
                'p99_ms': _quantile(vals, .99), 'mean_ms': _mean(vals)}
    def per100(key):
        vals = [float(x[key]) for x in runtime if x.get(key) is not None]
        return float(np.mean(vals) * 100.) if vals else None
    return {'status': 'MEASURED', 'scope': runtime[0].get('scope', 'UNDECLARED'),
            'latency_ms': {k: stats(k) for k in ('total_ms', 'source_ms', 'reference_ms', 'feature_ms', 'model_ms')},
            'per_100_plans': {k: per100(k) for k in ('source_rollout_calls', 'planning_source_rollout_calls',
                                                       'reference_rollout_calls', 'callback_count')},
            'planning_candidate_count_per_scene': _mean([float(x['planning_candidate_count']) for x in runtime if x.get('planning_candidate_count') is not None]),
            'reuse_fraction': _mean([float(x['reused']) for x in runtime if x.get('reused') is not None]),
            'fallback_fraction': _mean([float(x['fallback']) for x in runtime if x.get('fallback') is not None]),
            'warning': 'Collection timing is not online planning latency unless the transcript was captured at the planning callback boundary.'}


def extended_audit(rows: Iterable[dict[str, Any]], prediction_key: str = 'predicted',
                   contrast_threshold: float = .20, detection_threshold: float = .05,
                   bootstrap: int = 2000, seed: int = 51,
                   reuse_clearance_multiplier: float | None = None,
                   reuse_guard_multiplier: float | None = None):
    rows = list(rows)
    return {
        'schema': 'CBC_EXTENDED_AUDIT_V1',
        'records': len(rows),
        'yield': yield_summary(rows),
        'critical_pairs': critical_pair_metrics(
            rows, prediction_key=prediction_key,
            contrast_threshold=contrast_threshold,
            detection_threshold=detection_threshold,
            bootstrap=bootstrap, seed=seed),
        'false_safe_anchor_reuse': false_safe_reuse_summary(
            rows, reuse_clearance_multiplier, reuse_guard_multiplier, bootstrap=bootstrap, seed=seed),
        'event_timing': event_timing_summary(rows),
        'rare_risk_calibration': stratified_reference_calibration(rows, prediction_key),
        'runtime': runtime_summary(rows),
        'cost_rollup': runtime_summary(rows),
        'limits': [
            'Reference outcomes are finite-MC simulator conditional means.',
            'Legal route is unknown when lawful_route_match is NOT_EVALUATED.',
            'A guard flip summary is not event detector recall or lead time.',
            'Cluster bootstrap covers scene dependence, not unseen response-law shift.',
        ],
    }

