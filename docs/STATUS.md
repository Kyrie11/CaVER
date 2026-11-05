> **2026-11-05 CBC-first v2 提醒：本文件为上一版历史记录；最新版运行/评估以 [`CBC_FIRST_RUN_CN.md`](CBC_FIRST_RUN_CN.md) 与 [`CBC_V2_CODE_AUDIT_CN.md`](CBC_V2_CODE_AUDIT_CN.md) 为准。新的 `scripts/server_pipeline.sh` 是限额压缩 WOMD 的 CBC-first 脚本，不再导出无界的 `womd_training.jsonl`，也不联合宣布 DSE/TIV 已认证。**

# Implementation status and non-negotiable publication gates

**2026-11-04 NEW AUDIT OVERRIDES:** This older status table describes the original prototype. Read `AUDIT_REVISED_CN.md` and `EXECUTION_REVISED_CN.md` first. CBC now has a joint ego/union head and a standalone hybrid guard screen, and DSE now uses a plausible-safe LP regret rather than the older gated-only regret. The guard screen is NOT integrated into native reactive inference, the full delayed GNN is missing, and official end-to-end benchmark runs remain unverified. **

This checklist distinguishes an implemented/runnable component from an externally verified model or safety guarantee. It is written as research engineering documentation, not as claimed experimental results.

| Component | Implemented now | Tested locally | Remaining to claim paper-level reproduction |
|---|---|---|---|
| WOMD Scenario proto decoding | Yes, official schema parser and map feature extraction | Not on real WOMD shards: dataset unavailable here | Decode v1.3.1 training/validation shards on user's server; round-trip and inspect all 91 steps; check 20s separately |
| WOMD tf.Example decoding | Yes, explicit feature map and future validity | Not on real shards: TensorFlow unavailable here | Validate 128-object and future shape against official sample and Waymax loader output |
| WOMD observed-policy pretraining | Yes, history-only 10 Hz acceleration regression | Network forward and synthetic tests | Fit on actual WOMD, inspect the learned acceleration distribution, trainval loss, mode mismatch |
| Waymax bridge | Native WOMD->`SimulatorState` read adapter | Cannot test locally: Waymax unavailable | Test installed Waymax version and implement/run native ego-controlled IDM/sim-agent closed loop; audit agent-agent collisions |
| nuPlan bridge | DB read-only schema/inventory; official scenario extraction path | Inventory on missing path; no real DB | Validate DB table structure and maps against server; complete official closed-loop simulation |
| ReactSim-Bench bridge | Behavior inventory, provenance-checked external result JSONL | Inventory and malformed-export tests | Run official preprocessing and model checkpoints; validate real output mapping into schema |
| CARLA bridge | Synchronous 4-actor scenario runner, delayed observations, collision sensors | Syntax and unit checks only | Execute on CARLA server; inspect spawn legality and initial state parity with frozen experiment; test sensor association |
| Four-car RQ1 surrogate | Heterogeneous delay-following model; frozen searched positive; B mediator swap | **Yes**: frozen case and swap verified | CARLA physical replication; multiple unseen-reactor tests; naturalistic parameter range and frequency |
| CBC physics-anchored risk | Torch learnable residual risk, clearance, stopping slack; pair contrast; event features | **Yes**: forward/backward, offline fit | Replace source joint rollout with trained multi-agent SMART or equivalent; implement full causal-time graph residual and candidate reuse bound calibration |
| CBC response tubes | Nonnegative delayed-gain upper-bound recursion; separate physical/statistical paths | **Yes**: deterministic invariants | Derive genuinely validated gains/delay/mode envelopes in specific vehicle classes; integrate gating into inference |
| DSE decision gate | Whole-scene marginal coverage calibration, MC audit uncertainty, robust outer bounds and regret upper bound | **Yes**: unit and synthetic trial | Field-relevant reference-law calibration; real-prefix shield; source shift tests; prove full adaptive release-state coverage after TIV |
| TIV targeted query proposal | Typed candidate/pair/chain/window/source query; multiple simulator audit sources | **Yes**: simulated query execution | Online verified posterior update with pre-declared stopping and independent release calibration; separate truth model from query models |
| TIV acquisition | Torch regression head for signed simulated decision loss reduction | **Yes**: local offline query audit | Train/evaluate with real independent reactors; compare equal-budget entropy/EPG/random/full-query baselines |
| Real-world causal identification | No | N/A | The real joint potential-outcome law is not identified by observation alone. Report only partial bounds/model-conditional claims |

## Explicit blocked gates

1. **External data validation is blocked in this environment.** `/data0/senzeyu2/...` does not exist here. It exists only on the user's server. The current tests cannot establish its real file structure or dimensional correctness. `womd-check --decode` is mandatory.
2. **No full official SMART checkpoint or native Waymax reactive-policy end-to-end reproduction is implemented.** The trainable CBC head is a physically anchored reference architecture; its event feature construction is temporal but it is **not** the complete paper's causal-time graph residual network.
3. **Neither nuPlan nor ReactSim benchmark has been executed.** The latter's `custom_behaviors` PKLs alone are not native rollout results.
4. **CARLA implementation is an exploration runner, not a verified reproduction** of the frozen surrogate geometry or IDM parameters. Do not claim RQ1 CARLA replication until matching initial states/controls and collision audit are independently checked.
5. **DSE outputs may be UNKNOWN for every scenario.** Limited calibration scenes, missing driver policies or narrow tolerances make this the appropriate scientific outcome. Do not tune risk thresholds on the held-out test data to create accepted examples.
6. **No unconditional action-wise safety certificate is claimed.** All conformal claims require frozen predictors, disjoint exchangeable calibration scenes, a meaningful reference law and valid conditions after query. External distribution shift violates those conditions.
7. **No measurement of compute-budget advantage is claimed.** TIV query costs in code are toy unit costs for the acquisition example; real wall-clock synchronization, GPU batching and p95/p99 latency remain to be measured.

## Recommended next engineering changes

- Build a version-tested native Waymax closed-loop runner with different SimAgentActor/IDM policies and measure per-role agent-agent collision events on paired interventions.
- Train and evaluate a *full delayed message-passing CBC residual* against full candidate-conditioned multi-agent responses and safety margin labels. The current CBC head is a clean baseline, not the end of method implementation.
- Calibrate a freeze-then-select candidate set under unseen reactor families; train TIV on source-specific rollout audits with independent ground-truth-like reference policy families; separately calibrate final release states.
- Map matching for legal ego candidate generation (WOMD vector map / nuPlan map API) rather than the current geometric lane filter. Audit each scenario's ego candidate dynamic feasibility and route progression.
- Run CARLA with paired-world initialization and controlled mediator replay, then independently compare results to nominal four-car surrogate and unknown policy envelopes.

## RIDE reuse / exclusion

Audited `ride/geometry.py` inspired the OBB separating-axis geometry; the new functions and collision semantics are in `caver/sim/geometry.py`. RIDE's read-only ReactSim token auditing motivated `caver/integrations/reactsim_tokens.py`. We did **not** copy the known exploratory brake-command-only lag controller or the RIDE historical v-series scripts. The uploaded RIDE archive has no `ride/fourcar/` package, no `scripts/48_fourcar_mechanism.sh` and no matching frozen config files. Instead the `configs/fourcar_challenge.json` is a separately searched and identified synthetic fixture, with a new, openly identified result.
