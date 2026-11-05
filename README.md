# CaVER — CBC-first v2 research harness (2026-11-05)

**Start here: [`docs/CBC_FIRST_RUN_CN.md`](docs/CBC_FIRST_RUN_CN.md)** — 中文详尽指令，CARLA 独立 RPC 检查、WOMD 限额压缩导出、CBC 同意图边界风险排序消融、报告打包与真实物理/统计证据缺口。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
MODE=pilot bash scripts/cbc_synthetic_study.sh
# After real WOMD Scenario dependencies are present:
MODE=pilot bash scripts/cbc_womd_pipeline.sh
# Synthetic fourcar does not use Carla server:
python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge.json
python -m caver.cli carla-check --host 127.0.0.1 --port 2000 --output outputs/carla_check.json
```

**Not a deployable real-road controller or complete official-reactive closed-loop reproduction.** In particular, CBC learned delayed graph is NOT the paper's independently verified guard-error envelope; DSE and TIV terminal after-query certification remain research gaps.

See [`docs/CBC_V2_CODE_AUDIT_CN.md`](docs/CBC_V2_CODE_AUDIT_CN.md) for the code/paper audit. Legacy documents are kept for historical context and do not supersede v2 instructions.

---

# CaVER — 2026-11-04 audited research prototype

**不是可部署自动驾驶安全控制器，也不是已经完成论文全部方法/实验的最终复现。**

本代码由上传的 `CaVER.zip` 修订，按用户最新 `interactive planning.tex` 审核 CBC → DSE → TIV。最重要的修复是：WOMD 当前历史截断与未来标签隔离、延迟响应器真实历史、CBC joint union 风险与事件裕度有效掩码、DSE `Gamma_R` / `Gamma_P` 联合 LP、train/calibration/test 场景防泄露以及 TIV 不能跳过独立终端校准。

## 先看完整说明

- [`docs/EXECUTION_REVISED_CN.md`](docs/EXECUTION_REVISED_CN.md)：**从空环境到服务器训练/数据预检、ReactSim 官方下载和 CARLA 指令**。
- [`docs/AUDIT_REVISED_CN.md`](docs/AUDIT_REVISED_CN.md)：**论文与代码逐项对照、尚缺组件、CBC/DSE/TIV 指标与消融/证伪实验**。
- [`docs/STATUS.md`](docs/STATUS.md)：阻塞性问题与尚未验证的场景。
- [`docs/REACTSIM_EXPORT_SCHEMA.md`](docs/REACTSIM_EXPORT_SCHEMA.md)：外部官方 reactive rollout 的严格来源交换格式。

## 快速运行（**只验证合成研究原型**）

```bash
cd CaVER
python -m pip install -e '.[test]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
bash scripts/smoke.sh
python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge.json
```

## 数据目录（不修改用户已有数据库）

- WOMD v1.3.1：`/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/{training,validation}`；`tf_example` **仅 Waymax 可选**。
- nuPlan DB：`/data0/senzeyu2/dataset/nuplan/data/cache/public_set_*`；地图：`/data0/senzeyu2/dataset/nuplan/maps`。
- ReactSim：`/data0/senzeyu2/dataset/ReactSim-Bench/{repo,ckpts,data/custom_behaviors}`；**尚未下载时**：`bash scripts/download_reactsim.sh`，随后单独 Python 3.9 环境 `bash scripts/reactsim_preprocess.sh`。
- CARLA：已安装的 0.9.16 服务器由用户启动，再运行 `python -m caver.cli carla-run ...`；探索性先导 runner，非正式认证。

```bash
bash scripts/verify_womd.sh
# 仅在真实 WOMD 解码成功、机器资源足够时执行：
bash scripts/server_pipeline.sh
```

**不要在发布图表中把合成反应器参考标签误称为 WOMD 已观测的 off-log 候选真实因果结果。** 不要在 validation/test 上对 CBC、observational policy 或 DSE 阈值进行拟合。不下载 `tf_example` 不影响本版 Scenario 主流程。所有自动测试均无法代替用户服务器的真实 shards/API 和官方模型闭环测试。

**Corrected geometry:** simulator `[x,y,speed,yaw,length,width]` is explicitly reordered before OBB test. All collision distances from the previous ZIP are invalid and must be recomputed. This revision passed 40 local unit tests; the full smoke script timed out locally after its calibration step and remains to be rerun.

A reduced post-calibration test with 2 held-out synthetic episodes completed the split audit, RQ audit, DSE evaluate, and TIV audit; it yielded no critical pair contrasts and no certified/released action (0/2). See `docs/VALIDATION_RESULTS_CN.md`.
