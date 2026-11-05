# CaVER CBC v4 code audit

## Reproducible findings

1. `scripts/cbc_train_dual_gpu.sh` accepts an optional fourth `DEV_RECORDS`; the v4 invocation supplied only train/test/output, so `DEV=$TRAIN`. It prints a warning but still runs `cbc-boundary-study --development "$TRAIN"`. Therefore thresholds are fit on training records, not an independent calibration split. The resulting boundary study is exploratory and must not be used as calibration evidence.
2. Record timing is explicitly offline collection timing (`runtime.scope=OFFLINE_RECORD_COLLECTION_WALLCLOCK_NOT_ONLINE_PLANNING_LATENCY`). It measures source/reference rollouts, feature extraction and model calls, but `callback_count`, `reused`, and `fallback` are always null. It cannot support online p95/p99 or callback/rollout claims.
3. For each candidate, source rollouts are `sample_world(seed+k, source_family)` and reference rollouts are `sample_world(100000+seed+k, audit_family)`. Candidate actions are paired within each family but source and audit laws differ by construction. This is a simulator conditional transport test, not a real counterfactual label.
4. `candidate_matching.lawful_route_match` is hard-coded `NOT_EVALUATED`; `matched_kinematic_proxy` is a geometry/kinematics proxy only. `yield_summary` correctly leaves legal as unknown, but any aggregate that treats this as matched+legal is invalid.
5. `source_temporal_graph` uses source rollout means and current-geometry heuristic edges. Delay is a constant node channel; graph message passing is lag-shifted, then `GraphCascadeRiskHead` averages all time (`h.mean(dim=0)`). No measured online latency enters the graph or gate. `anchor_view` copies all anchor source pair event/clearance features across candidates and changes only action controls plus the ego node; this is a deliberate one-anchor reuse baseline, not candidate-conditioned response evidence.
6. Graph training predicts pair risk through `m.graph_model`, but `record_with_new_predictions` computes ego/downstream through `predict_joint(model, feat)`, which uses `m.joint` and does not see graph traces. Graph effects reach downstream risk only indirectly via `coherent_project`; report graph union risk as a joint-head/projection result, not a graph-conditioned union predictor.
7. `no_events` masks stop slack, valid fraction, brake onsets and path hops; pair contact and clearance remain. The no-events ablation is a capacity-matched event-mask ablation, but it does not isolate latency alone.
8. Extended audit definitions are finite Monte-Carlo simulator metrics: critical pair `|reference delta|>=0.2`, reference resolution uses a Hoeffding radius, underresolution is `|pred delta|<0.05`, sign error excludes underresolved predictions. Scene bootstrap is appropriate for pair dependence but only 3 resolved scenes in the uploaded test report, giving very wide CIs.

## Uploaded v4 evidence

`extended_audit.json`: 23 critical pair rows, only 3 reference-resolved; reference-resolved sign-error 0 for source/full/anchor variants except source 0.833 and anchor-copy 1.0 in method table; full and no_events tie (event gain absent); anchor_graph has pointwise improvement but scene CI [-1,0], inconclusive. Boundary screen uses 240 held-out pair comparisons, 59 flips, reuse 162, missed 2; CP upper miss 0.1029, so no 5% miss guarantee. Source rollout count reduction proxy is 66.7%, not wall-clock.

## Recommended code/protocol changes

- Require an explicit calibration/dev file for final boundary study (or label the result exploratory and omit it from the final gate). Generate a separate `--split calibration` record file and run `cbc_train_dual_gpu.sh train test out calibration`.
- Add a real planner-callback benchmark transcript with candidate count, predictor calls, callback wall-clock, cache hits, fallbacks, p50/p95/p99 total latency. Keep offline collection timing separate.
- Add route/map legality and ego-safety evidence before claiming matched+legal+ego-safe yield.
- Make graph head return a graph-conditioned joint ego/downstream head, or explicitly report downstream as joint-head output; do not imply graph explains union-risk improvement.
- For latency design, expose per-node delay distributions/quantiles and delayed event-boundary features, preserve temporal pooling/attention instead of global mean alone, and make REUSE/RECOMPUTE/ABSTAIN use measured callback latency plus conservative response envelope.
- Preserve source vs audit-family provenance in all reports and never call audit labels “ground truth.”
