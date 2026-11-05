> **2026-11-05 CBC-first v2 提醒：本文件为上一版历史记录；最新版运行/评估以 [`CBC_FIRST_RUN_CN.md`](CBC_FIRST_RUN_CN.md) 与 [`CBC_V2_CODE_AUDIT_CN.md`](CBC_V2_CODE_AUDIT_CN.md) 为准。新的 `scripts/server_pipeline.sh` 是限额压缩 WOMD 的 CBC-first 脚本，不再导出无界的 `womd_training.jsonl`，也不联合宣布 DSE/TIV 已认证。**

# 修订版本地验证记录（2026-11-04）

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`: **40 passed, 2.62s**。均为本地测试；官方 WOMD、nuPlan、CARLA、ReactSim 服务端接口未在该环境验证。
- `python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge_corrected.json`: 通过。修订几何下 Early B–C 接触，SAT signed margin −1.800m；Late 不接触 +1.509m；Ego–B 分别 +2.859m/+5.601m，两个 Ego 轨迹最大坐标距离 2.772m。B 结构性强制轨迹交换的 B–C 二值接触随之翻转。**案例为后验搜索并在延迟修复后调整参数的合成数据**；不是 WOMD 发现的自然事件或 CARLA 重现。
- `bash scripts/smoke.sh`：这一轮执行直到 `calibrate` 成功，输出 **n=6, quantile=1.0, mc_buffer≈0.826**。执行被本地单次 120 秒限制终止，尚未在该轮完成 test collect、split audit、RQ audit、DSE evaluate、TIV audit。之前其他修订阶段有过合成全流程输出，但**不得据此声称最终版本的所有阶段已验收**。
- 缩小版后续链路也已通过：`collect --split test --scenes 2 --source-worlds 1 --audit-worlds 1`；基于当前 train=5、cal=6、test=2 文件运行 `audit-splits` 无 scene ID/group 交叉；`rq-audit` 可执行但 `critical_pair_tests=0`，不支持任何 CBC 风险反转优势结论；`evaluate` 得到 2/2 `UNCERTIFIED`、`risk_admissible_scene_coverage=0`、`decision_certified_scene_coverage=0`；`tiv-audit --epochs 1` 得到 20 条合成查询代理训练记录。测试预算极小、完全合成，只用于功能验证。
- `WOMD/nuPlan/ReactSim`：当前文件系统没有用户服务器 `/data0` 下的真实数据；仅完成代码接口静态检查与单测模拟。外部模型、合法地图目标、真实驾驶仿真闭环及查询后认证均未完成。

若正式实验使用上面四车案例，应标注 synthetic posthoc stress-test，报告搜参程序/种子和未通过机制测试的负例；禁止把其碰撞频率当作自然交通统计量。
