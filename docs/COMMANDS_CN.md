> **2026-11-05 CBC-first v2 提醒：本文件为上一版历史记录；最新版运行/评估以 [`CBC_FIRST_RUN_CN.md`](CBC_FIRST_RUN_CN.md) 与 [`CBC_V2_CODE_AUDIT_CN.md`](CBC_V2_CODE_AUDIT_CN.md) 为准。新的 `scripts/server_pipeline.sh` 是限额压缩 WOMD 的 CBC-first 脚本，不再导出无界的 `womd_training.jsonl`，也不联合宣布 DSE/TIV 已认证。**

# 执行命令：修订版

旧版该文件使用 WOMD validation 同时构建 calibration 和 test，存在跨集合重复风险，**不再使用**。

请以 [`EXECUTION_REVISED_CN.md`](EXECUTION_REVISED_CN.md) 中的命令为准。端到端本地合成测试为 `bash scripts/smoke.sh`；服务器 Scenario 主流程为 `bash scripts/server_pipeline.sh`；ReactSim 下载为 `bash scripts/download_reactsim.sh`。所有指令的运行前提、依赖隔离和不能宣称的闭环结论都在新手册中。
