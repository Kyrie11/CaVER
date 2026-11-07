# CBC implementation audit and follow-up protocol

## Implementation alignment

- `DelayedGraphEncoder` consumes six source-only trace channels. Each message is shifted by a strictly positive lag, normalized by the kernel mass that has arrived, then combined with the receiver state and its previous time step through a two-layer SiLU update and LayerNorm.
- The joint ego/downstream head receives the graph summary when a graph variant is active. The summary is the concatenation of actor-wise temporal mean and max pools, with 32 channels at the default 16-channel graph width.
- The paper now uses a shared message matrix `W^(r)` because the implementation does not learn an independent matrix for every edge. Edge-specific timing remains in the response-arrival kernel.

## Code changes

1. The source edge screen scans the complete source-predicted horizon. It keeps the original coarse longitudinal/lateral envelope and records that the screen is heuristic. This prevents a future lane entry from being dropped solely because the agents were separated at the initial frame.
2. Anchor delay-kernel traces now compute the first coarse interaction-corridor time for the candidate ego trace and the anchor trace. Ego-originating kernels are shifted by the candidate-minus-anchor time difference and renormalized after duplicate lag bins are merged. Missing either encounter keeps the original kernel and records an unresolved shift.
3. CBC training exposes `contrast_threshold`, `rank_threshold`, and `reference_alpha`. The values are stored in checkpoint metadata. The default thresholds remain `0.20`, preserving the previous supervision contract.
4. The dual-GPU script trains an equal-budget `anchor_graph_delay_kernel` versus `delay_kernel_rank` pair. The only intended difference is the scene-level downstream-union ranking term. The script defaults all other variants to zero ranking weight and lets the rank treatment be set with `RANK_TREATMENT_WEIGHT`.
5. Checkpoint metadata records whether candidate arrival-kernel shifting was present in the training records. Equal-budget audits require this flag to match when comparing two treatments.
6. DSE calibration keeps separate one-sided episode maxima for underprediction and overprediction. The total calibration error is split equally between the two tails before allocation across ego, downstream and pair kinds. Upper risk gates use the former, while regret intervals retain the latter. Older symmetric calibration JSON remains loadable through the compatibility fields.
7. TIV's new value head predicts a signed mean and a positive scale. Query selection uses a predeclared lower-confidence value with `kappa=1.64`. Legacy scalar checkpoints still load through a compatibility adapter and do not receive an uncertainty benefit.

## Commands for the threshold sensitivity and ranking ablation

```bash
cd work/optimized_code
export CONTRAST_THRESHOLD=0.05
export RANK_THRESHOLD=0.05
export REFERENCE_ALPHA=0.05
export RANKING_DETECT=0.01
export BASE_RANK_WEIGHT=0.0
export RANK_CONTROL_WEIGHT=0.0
export RANK_TREATMENT_WEIGHT=0.10
bash scripts/cbc_train_dual_gpu.sh TRAIN_RECORDS TEST_RECORDS OUTPUT_DIR CALIBRATION_RECORDS
```

The resulting `cbc_study.json` compares `anchor_graph_delay_kernel` and `delay_kernel_rank` under the same scenes, optimizer updates, seed, and active architecture. It reports the original critical gate and a separate subcritical ranking diagnostic. Keep the original `.20` threshold run as the primary falsification condition. The `.05` condition is a development sensitivity condition and must never be merged into the critical certificate denominator. With three screened pairs and `alpha=.05`, an audit budget near 1024 worlds per reference-family aggregate is needed for a finite-MC radius near `0.10`; at 192 worlds the radius is about `0.22`, so lowering `tau` alone does not create resolved labels. Use at least 2048 worlds per reference family for a `.05` ranking threshold, then verify `ranking_scenes_per_epoch` and `ranking_pairs_per_epoch` in every training log.

The source edge screen and candidate kernel shift change the temporal evidence seen by graph variants. All CBC checkpoints using these changes require fresh training and held-out evaluation. Existing checkpoints remain loadable, but their metadata does not certify that they used the updated graph construction.

## DSE and TIV audit status

DSE remains a calibrated reference-law risk gate and plausible-safe regret test. Its upper gate is calibrated from one-sided underprediction residuals, while lower endpoints for regret use the opposite residual tail. It does not treat the graph output as a certificate. TIV remains a source-aware query proposer. Its probabilistic signed-value head and lower-confidence selection rule are offline simulator-conditional acquisition tools and cannot authorize a release. Positive-value and action-change rates must be nonzero on a held-out near-tie query audit before any planning-utility claim.
