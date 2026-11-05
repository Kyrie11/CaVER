"""Extended, non-destructive CBC audit metrics.

This module deliberately consumes the existing CaVER record dictionaries and
never changes the training or inference API.  It separates three quantities
that were previously conflated by the study code:

* ranking sign error (a resolved prediction points in the wrong direction),
* under-resolution (the prediction does not separate the candidates), and
* false-safe reuse (a reuse decision accepts a pair whose held-out risk is high).

The module is conservative about evidence.  If records do not contain a real
wall-clock trace, legal labels, event onsets, or explicit reuse decisions, the
corresponding report is returned with ``status='NOT_MEASURED'`` instead of
silently inferring a value from rollout counts or aggregate diagnostics.

The functions accept plain JSON-compatible dictionaries, so they can be run on
existing ``read_records`` output without importing torch.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NOT_MEASURED = "NOT_MEASURED"


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if isfinite(x) else None


def _get(d: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present key (including nested dotted keys)."""
    for key in keys:
        if key in d:
            return d[key]
        cur: Any = d
        ok = True
        for part in key.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok:
            return cur
    return None


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    a = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.quantile(a, q / 100.0)) for q in (50, 95, 99)}


def _sign(x: float, threshold: float = 0.0) -> int:
    return 1 if x > threshold else -1 if x < -threshold else 0


def _row_actions(row: Mapping[str, Any]) -> list[str]:
    return [str(a.get("action_id")) for a in row.get("actions", []) if isinstance(a, Mapping)]


def _matched(row: Mapping[str, Any], pair_key: str = "early|late") -> bool:
    m = _get(row, "candidate_matching.pairs." + pair_key)
    return bool(isinstance(m, Mapping) and m.get("matched_kinematic_proxy", False))


def _legal_status(row: Mapping[str, Any], pair_key: str = "early|late") -> bool | None:
    """Parse only explicit legal labels; ``NOT_EVALUATED`` is unknown."""
    m = _get(row, "candidate_matching.pairs." + pair_key)
    if not isinstance(m, Mapping):
        return None
    value = m.get("lawful_route_match", m.get("legal"))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"LEGAL", "VERIFIED", "TRUE", "YES", "PASS", "LAWFUL"}:
            return True
        if token in {"ILLEGAL", "FALSE", "NO", "FAIL", "UNLAWFUL"}:
            return False
    return None


def _ego_safe(row: Mapping[str, Any], actions: Iterable[str], threshold: float = 0.05) -> bool | None:
    vals: list[float] = []
    aud = row.get("audited", {})
    for action in actions:
        x = _num(aud.get(f"{action}/ego")) if isinstance(aud, Mapping) else None
        if x is None:
            return None
        vals.append(x)
    return bool(vals) and all(x <= threshold for x in vals)


def matched_legal_ego_safe_yield(
    records: Iterable[Mapping[str, Any]],
    *,
    pair_key: str = "early|late",
    ego_risk_threshold: float = 0.05,
) -> dict[str, Any]:
    """Report the candidate-pair yield at each evidence gate.

    The legal gate is *unknown* when the record contains the historical
    ``NOT_EVALUATED`` marker.  Its denominator and yield are therefore not
    fabricated.  Matched and ego-safe yields remain available independently.
    """
    rows = list(records)
    matched_n = legal_n = ego_safe_n = 0
    legal_known_n = ego_known_n = 0
    for row in rows:
        if not _matched(row, pair_key):
            continue
        matched_n += 1
        legal = _legal_status(row, pair_key)
        if legal is not None:
            legal_known_n += 1
            if legal:
                legal_n += 1
        actions = pair_key.split("|")
        safe = _ego_safe(row, actions, ego_risk_threshold)
        if safe is not None:
            ego_known_n += 1
            if safe:
                ego_safe_n += 1
    total = len(rows)
    status = "MEASURED" if legal_known_n else NOT_MEASURED
    return {
        "status": status,
        "total_scenes": total,
        "matched_scenes": matched_n,
        "matched_yield": matched_n / total if total else None,
        "legal_known_matched_scenes": legal_known_n,
        "legal_matched_scenes": legal_n if legal_known_n else None,
        "legal_given_matched_yield": legal_n / legal_known_n if legal_known_n else None,
        "matched_and_legal_scenes": legal_n if legal_known_n else None,
        "ego_safe_known_matched_scenes": ego_known_n,
        "ego_safe_scenes": ego_safe_n if ego_known_n else None,
        "ego_safe_given_matched_yield": ego_safe_n / ego_known_n if ego_known_n else None,
        "matched_legal_ego_safe_scenes": (
            sum(1 for row in rows
                if _matched(row, pair_key)
                and _legal_status(row, pair_key) is True
                and _ego_safe(row, pair_key.split("|"), ego_risk_threshold) is True)
            if legal_known_n and ego_known_n else None
        ),
        "definition": "matched -> explicit legal label -> both candidates ego-safe",
        "legal_warning": "lawful_route_match=NOT_EVALUATED is not treated as legal",
    }


def _reference_radius(row: Mapping[str, Any], alpha: float = 0.05) -> float | None:
    """Conservative Hoeffding radius for a difference of two Bernoulli means."""
    n = _num(row.get("reference_worlds_per_candidate"))
    if n is None:
        fam = row.get("audit_families", [])
        worlds = _num(row.get("audit_worlds_per_family"))
        if worlds is not None:
            n = len(fam) * worlds
    if n is None or n <= 0:
        return None
    # Each action mean has radius sqrt(log(4/alpha)/(2n)); sum both radii.
    return float(2.0 * sqrt(np.log(4.0 / alpha) / (2.0 * n)))


def _pair_records(
    row: Mapping[str, Any],
    *,
    early: str,
    late: str,
    contrast_threshold: float,
    detect_threshold: float,
    only_matched: bool,
    alpha: float,
) -> list[dict[str, Any]]:
    if only_matched and not _matched(row, pair_key=f"{early}|{late}"):
        return []
    pred = row.get("predicted", {})
    aud = row.get("audited", {})
    radius = _reference_radius(row, alpha=alpha)
    out: list[dict[str, Any]] = []
    for pair in row.get("pairs", []):
        tk = f"{early}/{pair}"
        lk = f"{late}/{pair}"
        true_e, true_l = _num(aud.get(tk)), _num(aud.get(lk))
        pred_e, pred_l = _num(pred.get(tk)), _num(pred.get(lk))
        if None in (true_e, true_l, pred_e, pred_l):
            continue
        true_delta = true_e - true_l
        pred_delta = pred_e - pred_l
        if abs(true_delta) < contrast_threshold:
            continue
        pred_resolved = abs(pred_delta) >= detect_threshold
        true_sign = _sign(true_delta)
        pred_sign = _sign(pred_delta) if pred_resolved else 0
        out.append({
            "scene_id": row.get("scene_id"), "pair": pair,
            "true_delta": true_delta, "pred_delta": pred_delta,
            "reference_sign_resolved": (
                abs(true_delta) > radius if radius is not None else None),
            "predicted_resolved": pred_resolved,
            "sign_error": bool(pred_resolved and pred_sign != true_sign),
            "underresolution": bool(not pred_resolved),
            "false_safe_pair": bool(max(pred_e, pred_l) < detect_threshold and max(true_e, true_l) >= contrast_threshold),
            "reference_radius": radius,
        })
    return out


def critical_pair_audit(
    records: Iterable[Mapping[str, Any]],
    *,
    early: str = "early",
    late: str = "late",
    contrast_threshold: float = 0.20,
    detect_threshold: float = 0.05,
    only_matched: bool = True,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Separate sign errors, under-resolution, and critical misses.

    The paper's primary denominator is the reference-resolved critical-pair
    set.  All-critical rates are retained as supplementary diagnostics only.
    Confidence intervals resample scenes, keeping all pair rows from a scene
    together.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    rows = list(records)
    pairs = [p for row in rows for p in _pair_records(
        row, early=early, late=late, contrast_threshold=contrast_threshold,
        detect_threshold=detect_threshold, only_matched=only_matched, alpha=alpha)]
    resolved_ref = [p for p in pairs if p["reference_sign_resolved"] is True]
    unresolved_ref = [p for p in pairs if p["reference_sign_resolved"] is False]
    unknown_ref = [p for p in pairs if p["reference_sign_resolved"] is None]
    pred_resolved = [p for p in pairs if p["predicted_resolved"]]

    for pair in pairs:
        pair["critical_miss"] = bool(pair["sign_error"] or pair["underresolution"])

    scene_bootstrap = {
        "sign_error_rate_reference_resolved": _bootstrap_scene_rate(
            resolved_ref, "sign_error", n_boot=n_boot, seed=seed),
        "underresolution_rate_reference_resolved": _bootstrap_scene_rate(
            resolved_ref, "underresolution", n_boot=n_boot, seed=seed + 1),
        "critical_miss_rate_reference_resolved": _bootstrap_scene_rate(
            resolved_ref, "critical_miss", n_boot=n_boot, seed=seed + 2),
    }

    mean = lambda xs, key: float(np.mean([bool(x[key]) for x in xs])) if xs else None
    return {
        "status": "MEASURED" if pairs else NOT_MEASURED,
        "primary_metrics_status": "MEASURED" if resolved_ref else NOT_MEASURED,
        "definition": {
            "sign_error": "resolved prediction has opposite sign to reference delta",
            "underresolution": f"abs(predicted_delta) < {detect_threshold}",
            "critical_miss": "sign_error OR underresolution",
            "false_safe_pair": f"max(predicted risks)<{detect_threshold} and max(reference risks)>={contrast_threshold}",
            "critical_pair": f"abs(reference_delta)>={contrast_threshold}",
            "primary_denominator": "reference-resolved critical pairs: abs(reference_delta) > finite-reference MC radius",
            "confidence_interval": "scene-cluster bootstrap; pair rows within a scene are resampled together",
            "mc_alpha": alpha,
        },
        "critical_pair_tests": len(pairs),
        "reference_sign_resolved_pairs": len(resolved_ref),
        "reference_unresolved_pairs": len(unresolved_ref),
        "reference_resolution_unknown_pairs": len(unknown_ref),
        "predicted_resolved_pairs": len(pred_resolved),
        "sign_error_rate_all_critical": mean(pairs, "sign_error"),
        "sign_error_rate_reference_resolved": mean(resolved_ref, "sign_error"),
        # Primary paper metric: denominator is reference-resolved critical pairs.
        "underresolution_rate": mean(resolved_ref, "underresolution"),
        "critical_miss_rate_reference_resolved": mean(resolved_ref, "critical_miss"),
        # Supplementary rates preserve visibility into MC-unresolved critical pairs.
        "underresolution_rate_all_critical": mean(pairs, "underresolution"),
        "critical_miss_rate_all_critical": mean(pairs, "critical_miss"),
        "false_safe_pair_rate": mean(pairs, "false_safe_pair"),
        "scene_cluster_bootstrap": scene_bootstrap,
        "rows": pairs,
    }


def _bootstrap_scene_ratio(
    scene_counts: Sequence[tuple[int, int]], *, n_boot: int = 2000, seed: int = 17
) -> dict[str, Any]:
    if not scene_counts:
        return {"status": NOT_MEASURED, "point": None, "ci95": None, "n_scenes": 0}
    a = np.asarray(scene_counts, dtype=float)
    point = float(a[:, 0].sum() / a[:, 1].sum()) if a[:, 1].sum() else None
    if len(a) < 2 or point is None:
        return {"status": "MEASURED", "point": point, "ci95": None, "n_scenes": len(a)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = a[rng.integers(0, len(a), len(a))]
        den = sample[:, 1].sum()
        if den:
            vals.append(float(sample[:, 0].sum() / den))
    return {
        "status": "MEASURED", "point": point,
        "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else None,
        "n_scenes": len(a), "n_boot": len(vals), "cluster": "scene",
    }


def _bootstrap_scene_rate(
    items: Sequence[Mapping[str, Any]], key: str, *, n_boot: int = 2000, seed: int = 17
) -> dict[str, Any]:
    """Bootstrap a pair-level rate while resampling whole scene clusters."""
    if not items:
        return {"status": NOT_MEASURED, "point": None, "ci95": None, "n_scenes": 0}
    if any(item.get("scene_id") is None for item in items):
        missing = sum(item.get("scene_id") is None for item in items)
        return {
            "status": NOT_MEASURED, "point": float(np.mean([bool(x[key]) for x in items])),
            "ci95": None, "n_scenes": len({x.get("scene_id") for x in items if x.get("scene_id") is not None}),
            "reason": f"scene_id missing for {missing} reference-resolved critical pair rows",
        }
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_scene[str(item["scene_id"])].append(item)
    counts = [
        (sum(int(bool(item[key])) for item in group), len(group))
        for group in by_scene.values()
    ]
    return _bootstrap_scene_ratio(counts, n_boot=n_boot, seed=seed)


def reference_flip_audit(
    records: Iterable[Mapping[str, Any]], *,
    contact_flip_min: float = 0.0,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Return pair flip MC intervals and a scene-cluster bootstrap interval.

    A pair MC interval is emitted only when the record stores the number of
    paired latent worlds.  A float flip rate without its denominator is not
    reverse-engineered into a fake sample size.
    """
    rows = list(records)
    items: list[dict[str, Any]] = []
    scene_counts: list[tuple[int, int]] = []
    for row in rows:
        scene_flip = scene_total = 0
        for pair, d in (row.get("boundary_diagnostics", {}) or {}).items():
            ref = d.get("reference", {}) if isinstance(d, Mapping) else {}
            rate = _num(ref.get("contact_flip_rate"))
            if rate is None:
                continue
            n = _num(ref.get("n_paired_latent_worlds"))
            is_flip = bool(rate > contact_flip_min)
            scene_flip += int(is_flip)
            scene_total += 1
            item = {"scene_id": row.get("scene_id"), "pair": pair,
                    "flip_rate": rate, "flip": is_flip, "n": int(n) if n else None}
            if n and n >= 1:
                # CP interval uses the finite paired-world event count.
                k = int(round(rate * n))
                try:
                    from scipy.stats import beta  # type: ignore
                    item["mc_ci95"] = [
                        0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
                        1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)),
                    ]
                except Exception:
                    rad = sqrt(np.log(40.0) / (2.0 * n))
                    item["mc_ci95"] = [max(0.0, rate - rad), min(1.0, rate + rad)]
            else:
                item["mc_ci95"] = None
            items.append(item)
        if scene_total:
            scene_counts.append((scene_flip, scene_total))
    return {
        "status": "MEASURED" if items else NOT_MEASURED,
        "pair_count": len(items),
        "flip_count": sum(int(x["flip"]) for x in items),
        "flip_rate": (sum(int(x["flip"]) for x in items) / len(items)) if items else None,
        "scene_cluster_bootstrap": _bootstrap_scene_ratio(scene_counts, n_boot=n_boot, seed=seed),
        "pair_mc_ci_available": sum(x["mc_ci95"] is not None for x in items),
        "pairs": items,
    }


def _iter_event_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("event_diagnostics", "event_predictions", "events", "event_metrics"):
        value = row.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, Mapping)]
        if isinstance(value, Mapping):
            out = []
            for name, item in value.items():
                if isinstance(item, Mapping):
                    out.append({"event": name, **item})
            return out
    return []


def event_detection_audit(records: Iterable[Mapping[str, Any]], *, _include_strata: bool = True) -> dict[str, Any]:
    """Compute event miss/lead-time only from explicit event records."""
    events: list[Mapping[str, Any]] = [e for row in records for e in _iter_event_rows(row)]
    if not events:
        return {"status": NOT_MEASURED,
                "reason": "records contain no explicit reference/predicted event onset or detection fields"}
    tp = fp = fn = tn = 0
    lead: list[float] = []
    onset_abs: list[float] = []
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in events:
        ref_present = e.get("reference_present", e.get("ref_detected"))
        pred_present = e.get("predicted_present", e.get("pred_detected"))
        ref_present = bool(ref_present) if ref_present is not None else _num(e.get("reference_onset_s", e.get("ref_onset_s"))) is not None
        pred_present = bool(pred_present) if pred_present is not None else _num(e.get("predicted_onset_s", e.get("pred_onset_s"))) is not None
        if pred_present and ref_present: tp += 1
        elif pred_present: fp += 1
        elif ref_present: fn += 1
        else: tn += 1
        ro = _num(e.get("reference_onset_s", e.get("ref_onset_s")))
        po = _num(e.get("predicted_onset_s", e.get("pred_onset_s")))
        if ro is not None and po is not None and pred_present and ref_present:
            lead.append(ro - po)  # positive = prediction leads the reference
            onset_abs.append(abs(po - ro))
        strata[str(e.get("stratum", "all"))].append(e)
    result = {
        "status": "MEASURED", "events": len(events), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "miss_rate": fn / (tp + fn) if tp + fn else None,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "lead_time_s": _percentiles(lead),
        "onset_abs_error_s": _percentiles(onset_abs),
        "lead_time_definition": "reference_onset - predicted_onset; positive means early alarm",
        "censoring_policy": "events without explicit onset are counted through presence fields only",
    }
    if _include_strata:
        result["strata"] = {}
        for name, group in strata.items():
            sub = event_detection_audit([{"events": group}], _include_strata=False)
            result["strata"][name] = {k: v for k, v in sub.items() if k not in ("strata", "status")}
    return result


def _calibration_points(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in records:
        pred, aud = row.get("predicted", {}), row.get("audited", {})
        # ``reference`` is an attribution baseline, not a planner candidate;
        # exclude it from candidate calibration.  Explicitly classify ego,
        # downstream-union, and non-ego-pair risks instead of inferring kind
        # from arbitrary key suffixes.
        actions = {str(a.get("action_id")) for a in row.get("actions", [])
                   if isinstance(a, Mapping)} - {"reference"}
        pairs = {str(p) for p in row.get("pairs", [])}
        for action in sorted(actions):
            for suffix, kind in (("/ego", "ego"), ("/down", "downstream_union")):
                key = action + suffix
                p, y = _num(pred.get(key)), _num(aud.get(key))
                if p is not None and y is not None:
                    points.append({"prediction": p, "reference": y,
                                   "scene_id": row.get("scene_id"),
                                   "kind": kind, "key": key})
            for pair in sorted(pairs):
                key = action + "/" + pair
                p, y = _num(pred.get(key)), _num(aud.get(key))
                if p is not None and y is not None:
                    points.append({"prediction": p, "reference": y,
                                   "scene_id": row.get("scene_id"),
                                   "kind": "non_ego_pair", "key": key})
    return points


def rare_risk_calibration(
    records: Iterable[Mapping[str, Any]],
    *,
    bins: Sequence[float] = (0.0, .01, .05, .20, .50, 1.0),
) -> dict[str, Any]:
    """Risk-stratified reliability for finite-MC reference means.

    The default record schema stores conditional reference means, not binary
    outcomes.  Therefore the report calls the metric ``reference_mean_ece``
    and does not mislabel it as a binary Brier score.
    """
    points = _calibration_points(records)
    if not points:
        return {"status": NOT_MEASURED, "reason": "no predicted/audited risk pairs"}
    bins = tuple(float(x) for x in bins)
    strata = []
    for kind in ("ego", "downstream_union", "non_ego_pair"):
      kind_points = [x for x in points if x["kind"] == kind]
      for lo, hi in zip(bins[:-1], bins[1:]):
        group = [x for x in kind_points if (lo <= x["prediction"] < hi) or
                 (hi == bins[-1] and lo <= x["prediction"] <= hi)]
        n = len(group)
        if not n:
            strata.append({"kind": kind, "lower": lo, "upper": hi, "count": 0,
                           "mean_prediction": None, "mean_reference": None,
                           "abs_gap": None, "reference_mean_ece_contribution": None})
            continue
        mp = float(np.mean([x["prediction"] for x in group]))
        my = float(np.mean([x["reference"] for x in group]))
        strata.append({"kind": kind, "lower": lo, "upper": hi, "count": n,
                       "mean_prediction": mp, "mean_reference": my,
                       "abs_gap": abs(mp - my),
                       "reference_mean_ece_contribution": n / len(kind_points) * abs(mp - my) if kind_points else None,
                       "rare_risk": bool(hi <= .05)})
    ece_by_kind = {}
    for kind in ("ego", "downstream_union", "non_ego_pair"):
        ece_by_kind[kind] = float(sum((x["reference_mean_ece_contribution"] or 0.0)
                                      for x in strata if x["kind"] == kind))
    return {"status": "MEASURED", "points": len(points), "candidate_points": len(points), "bins": strata,
            "reference_mean_ece": ece_by_kind,
            "label": "finite-K surrogate reference mean; not binary ECE/Brier"}


def cost_rollup(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize measured timing/callbacks and separately expose declared counts."""
    rows = list(records)
    total_ms: list[float] = []
    callback: list[float] = []
    rollout: list[float] = []
    component: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        timing = row.get("runtime", row.get("timing", row.get("latency", {})))
        if not isinstance(timing, Mapping): timing = {}
        x = _num(_get(timing, "total_ms", "wall_time_ms", "measured_walltime_ms"))
        if x is not None: total_ms.append(x)
        c = _num(_get(row, "callback_count", "planning_callback_count", "runtime.callback_count", "timing.callback_count"))
        if c is not None: callback.append(c)
        r = _num(_get(row, "rollout_count", "source_rollout_count", "runtime.source_rollout_calls", "timing.rollout_count"))
        if r is not None: rollout.append(r)
        for name in ("anchor_rollout_ms", "feature_ms", "cbc_ms", "graph_ms", "query_ms", "fallback_ms"):
            y = _num(_get(timing, name))
            if y is not None: component[name].append(y)
    declared_source = []
    for row in rows:
        worlds = _num(row.get("source_worlds"))
        n_actions = len([a for a in _row_actions(row) if a != 'reference'])
        if worlds is not None and n_actions:
            declared_source.append(worlds * n_actions)
    result = {
        "status": "MEASURED" if total_ms else NOT_MEASURED,
        "measured_scene_count": len(total_ms),
        "total_wall_time_ms": _percentiles(total_ms),
        "component_wall_time_ms": {k: _percentiles(v) for k, v in component.items()},
        "callback_count": _percentiles(callback),
        "rollout_count": _percentiles(rollout),
        "rollout_calls_per_100_decisions": float(np.mean(rollout) * 100.0) if rollout else None,
        "declared_source_rollouts_per_scene": _percentiles(declared_source),
        "declared_source_rollouts_per_planning_candidates": _percentiles(declared_source),
        "declared_counts_are_not_wall_time": True,
        "missing_measurements": [k for k, v in {
            "total_wall_time_ms": total_ms, "callback_count": callback, "rollout_count": rollout
        }.items() if not v],
    }
    return result


def false_safe_reuse_audit(
    records: Iterable[Mapping[str, Any]], *,
    high_risk_threshold: float = .20,
    safe_threshold: float = .05,
    n_boot: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Audit explicit reuse decisions; never infer reuse from anchor_copy."""
    decisions: list[dict[str, Any]] = []
    for row in records:
        value = row.get("reuse_decisions", row.get("reuse"))
        if isinstance(value, Mapping): value = list(value.values())
        if not isinstance(value, list): continue
        for item in value:
            if not isinstance(item, Mapping): continue
            accepted = item.get("accepted", item.get("reused", item.get("reuse")))
            p = _num(item.get("predicted_risk", item.get("predicted")))
            y = _num(item.get("reference_risk", item.get("reference")))
            if accepted is None or p is None or y is None: continue
            decisions.append({"accepted": bool(accepted), "predicted": p, "reference": y,
                              "false_safe": bool(accepted and p < safe_threshold and y >= high_risk_threshold),
                              "stratum": str(item.get("stratum", "all")),
                              "scene_id": row.get("scene_id")})
    if not decisions:
        return {"status": NOT_MEASURED,
                "reason": "records contain no explicit reuse_decisions with predicted/reference risk"}
    accepted = [x for x in decisions if x["accepted"]]
    fs = [x for x in accepted if x["false_safe"]]
    return {"status": "MEASURED", "decisions": len(decisions), "accepted": len(accepted),
            "reuse_fraction": len(accepted) / len(decisions),
            "false_safe_count": len(fs),
            "false_safe_rate_among_reused": len(fs) / len(accepted) if accepted else None,
            "false_safe_rate_all_decisions": len(fs) / len(decisions),
            "strata": {
                s: {"decisions": len(g), "accepted": sum(x["accepted"] for x in g),
                    "false_safe": sum(x["false_safe"] for x in g)}
                for s in sorted({x["stratum"] for x in decisions})
                for g in [[y for y in decisions if y["stratum"] == s]]
            },
            "definition": f"accepted && predicted<{safe_threshold} && reference>={high_risk_threshold}"}


def extended_audit(records: Iterable[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Run all non-destructive extensions in one call."""
    rows = list(records)
    return {
        "critical_pairs": critical_pair_audit(rows, **kwargs.get("critical", {})),
        "reference_flips": reference_flip_audit(rows, **kwargs.get("flip", {})),
        "matched_legal_ego_safe_yield": matched_legal_ego_safe_yield(rows, **kwargs.get("yield", {})),
        "false_safe_reuse": false_safe_reuse_audit(rows, **kwargs.get("reuse", {})),
        "events": event_detection_audit(rows),
        "rare_risk_calibration": rare_risk_calibration(rows, **kwargs.get("calibration", {})),
        "cost": cost_rollup(rows),
    }

if __name__ == "__main__":
    import argparse
    import gzip
    import json

    parser = argparse.ArgumentParser(description="Run conservative CBC extended audits on JSONL records")
    parser.add_argument("records", help="JSONL or JSONL.gz file produced by CaVER")
    parser.add_argument("--out", required=True, help="Output JSON report")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Scene bootstrap resamples for critical-pair and reference-flip CIs")
    parser.add_argument("--seed", type=int, default=17,
                        help="Random seed for scene-cluster bootstrap intervals")
    args = parser.parse_args()
    opener = gzip.open if args.records.endswith(".gz") else open
    rows = []
    with opener(args.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    report = extended_audit(
        rows,
        critical={"n_boot": args.bootstrap, "seed": args.seed},
        flip={"n_boot": args.bootstrap, "seed": args.seed},
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({
        "records": len(rows),
        "out": args.out,
        "statuses": {k: v.get("status") for k, v in report.items()},
    }, ensure_ascii=False))
