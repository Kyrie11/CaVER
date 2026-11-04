# 执行命令：修订版

旧版该文件使用 WOMD validation 同时构建 calibration 和 test，存在跨集合重复风险，**不再使用**。

请以 [`EXECUTION_REVISED_CN.md`](EXECUTION_REVISED_CN.md) 中的命令为准。端到端本地合成测试为 `bash scripts/smoke.sh`；服务器 Scenario 主流程为 `bash scripts/server_pipeline.sh`；ReactSim 下载为 `bash scripts/download_reactsim.sh`。所有指令的运行前提、依赖隔离和不能宣称的闭环结论都在新手册中。
