# CaVER v2 源码审计摘要

依据最新 2026-11-05 上传 `.tex` 的 CBC / DSE / TIV 定义，以本次 `result.zip` 的 9 份报告为调试起点。

## 核心实现新增/修正

| 文件 | 新增逻辑 | 仍然欠缺 |
|---|---|---|
| `caver/data/preparation.py` | WOMD Scenario 流式训练/校准/测试分区、只保留当前+历史、两种受控视图、小于 45 MiB `.jsonl.gz`、manifest 记录来源 | 20s 原片段 ID 不在标准 Scenario 可直接验证的信息内；地图完整语义尚未加入候选过滤 |
| `caver/data/womd.py` | 随机化 shard 迭代顺序、限制分片/条数；原来的历史与未来标签读取分离 | 仅原生 `current_time_index`，不是全时间步 sequence 训练 |
| `caver/sim/traffic.py` | 不同结构响应器、延迟历史、候选 Ego open-loop action、接触与有效事件裕度 | 代理 IDM 不代表 CARLA/WOMD/nuPlan 原生 human policy |
| `caver/models/temporal.py` | 仅从 source reactor 预测提取严格过去到未来的延迟图，正时延约束、Monte Carlo 世界顺序无关 | 图是预测网络，不是**经证明**完整 event guard 集合 |
| `caver/models/anchor.py` | 只复制一个源 anchor 的响应证据，Late/Reference 使用候选 Ego 预定轨迹差分，禁止非 anchor source rollouts/审核标签进入 CBC inference | 非 anchor 交互响应由统计修正，物理认证的异常分支尚未完成 |
| `caver/models/cbc.py` | full / no-events / anchor-full / anchor-graph 风险头，独立第三方 union 事件头 | pair/union 后处理只保证必要概率关系，不提供联合分布的正确性证明 |
| `caver/experiment.py` | 同意图候选动力学/任务匹配，源/独立审计 Monte Carlo 配对、接触/刹车 guard 差分，支持 gzip | CARLA/ReactSim 的独立原生参考模型未替换合成世界 |
| `caver/validation/cbc_study.py` | 同意图有效样本、临界 pair sign error、reference 有限 K 可辨识性、near/far 误差、独立 scene bootstrap，对比 anchor-only/full/no-events | 样本充分性、真实合法道路和自然场景分布待独立确认 |
| `caver/validation/boundary_study.py` | 经验 guard 阈值独立 train 拟合/test 冻结，reuse vs missed flip 和 Clopper Pearson 上界 | **不能**代替 hybrid guard 物理扰动包络，online safety certificate 必须视为未知 |
| `caver/validation/cbc_gate.py` | 阻止小样本结果被解读为肯定结论 | 不是实际的统计显著性检验或外部闭环证书 |
| `caver/integrations/carla_runner.py` | 独立 RPC/version/map 预检查、碰撞传感器回调角色隔离、初始状态差异报告 | CARLA runner 内控制器不同于冻结四车合成世界、尚无角色随机化和大量初始条件统计 |
| `scripts/package_diagnostics.py` | 白名单报告打包，默认 <20 MiB、绝对禁止原始 records/checkpoint | 只适用于明确配置的本地输出文件 |
| `scripts/server_pipeline.sh` | 安全默认 CBC-first，禁用旧无界 training/validation JSONL 导出流程 | 旧联合 surrogate 流程保留为 `server_pipeline_legacy_do_not_use.sh`，仅供追溯 |

## 审计到的关键错误预防

- `fourcar-challenge` 永远只执行**合成**模型；不要将其报告来源解释成 CARLA import 错误。
- WOMD 的 `future` 在 Scenario proto 中客观存在，但被 `with_future=False` 的 data adapter 排除于输入 Scene；所有输出必须标记 `future_trajectory_exposed=false`。
- `corridor_subset()` 在 WOD 世界坐标转局部坐标后保留了有效历史轨迹；从 `womd-prep corridor` 读入的 scene **不可进行第二次**相同坐标转换，否则无法复现。
- `rollout()` 状态 `[x,y,speed,yaw,L,W]` 在 OBB 碰撞前必须切成 `[x,y,yaw,L,W]`，否则速度会误当航向角。
- 异常无刹车事件不能被虚构的固定正停车裕度当作有效标签。对应 valid mask 永不从参考未来泄露到预测输入。
- 训练 CBC 只能用 train；独立测试和校准记录需通过 ID/group 防交集检验。当前 group 为保守**启发式**，不能取代完整日志分割。
- 依据最新论文，`RISK-ADMISSIBLE` 与 `DECISION-CERTIFIED` 不等价；无完整物理前缀的部署不允许自动释放。

## 具体验证记录

- 本地 `pytest`：51 项通过（见运行终端；不包含服务器 WOMD/CARLA 官方验收）。
- 小规模完整 synthetic-stress 流程：6 train / 6 test，K=6 reference worlds/action；`source`、`anchor_copy`、`full`、`no_events`、`anchor_full`、`anchor_graph` 均完成风险审计；`critical_pair_tests=1`、`critical_pair_ref_sign_resolved=0`。当前结果是 **INSUFFICIENT_REFERENCE_EVIDENCE**，不能据此称任一方法有效。
- 样本不足时 `cbc-gate` 会输出缺口；有正向描述趋势也必须看 bootstrap CI 及独立模型域验证。
- **未在该隔离环境使用真实 1000/150 WOMD shards、CARLA 0.9.16 RPC、nuPlan 原生官方 devkit 或 ReactSim-Bench checkpoint 完成闭环**。其检查应由用户服务器按本手册执行。

**指标和执行顺序详见 `CBC_FIRST_RUN_CN.md`。**
