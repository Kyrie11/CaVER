# Canonical external ReactSim reactive rollout exchange

Generate one JSON record per `(scene_id, action_id, reactor)` by **executing the official ReactSim-Bench simulation**, not by reading the arbitrary custom behavior specification pickle and copying log trajectories. Mandatory fields:

```json
{"scene_id":"original_nuplan_token","action_id":"early","reactor":"official_model_id",
 "scene_digest":"sha256_original_initial_state","action_digest":"sha256_precommitted_ego_trajectory",
 "rollout_digest":"sha256_complete_reactive_multiagent_trace","ego_collision":false,
 "downstream_collision":true,"progress":32.5,"runtime_ms":63.2,
 "physical_contact_source":"sim_collision"}
```

`physical_contact_source` must be `sim_collision` (the official simulator) or `trajectory_swept_obb` (state explicitly that the result was derived from full trajectories rather than contact sensors). Never convert a precomputed behavior label into `downstream_collision` without validating the executed agent pair and future. For credible per-agent safety comparison, add all per-pair contacts and continuous trajectories, plus controlled ego policy, response model checkpoint ID, nuPlan base log identity, map version, random seed and executed horizon as additional fields.

Code: `caver.data.reactsim.validate_export` rejects absent provenance or duplicate action+scene+reactor identities. It does **not** claim to execute ReactSim or verify physics on its own.
