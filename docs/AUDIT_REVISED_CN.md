# CaVER 最新论文与旧 ZIP 工程的代码审计／实验路线（2026-11-04）

本审计以用户上传的 `interactive planning.tex` **最新修订版**为规范，以 `CaVER.zip` 的代码为待修工程。旧版 `代码设计.md` 描述的是之前模型的交付范围，不能替代最新论文定义。此报告区分“已修改并通过无真实数据的单测”“已完成合成试验”和“尚未完成真实场景/官方模拟器验证”。

## 一、论文机制 -> 代码状态的逐项映射

| 最新论文设计 | 本版代码路径 | 现在实际有的能力 | 仍缺什么，不能宣称已复现 |
|---|---|---|---|
| CBC 候选动作的配对响应结构潜变量 | `caver/sim/traffic.py` | 四车异质时延、按固定种子配对不同 ego 候选，反应由过去观测触发 | 独立的 SMART/ReactSim 响应主干，所有真实多车交互分布 |
| CBC 时延、制动 guard、例外集合 \(\mathfrak B\) | `caver/validation/boundary.py` | `AnchorWorld`/`VerifiedPerturbation`/`assess_boundary`：缺任何经过验证的 guard 或界时默认 `beta=1`、禁止安全复用；Hoeffding 事件质量上界、source-law transfer | **最关键缺项**：完整时变有向图 \(F_\theta,M_\theta\)、**经真实/独立物理模型认证**的增量误差界；此 boundary 函数目前**未连入**主 CBC 候选 rollout 的求解器，因此不能宣称已经实现可信在线快速复用 |
| CBC 接触、首次冲突停车裕度及概率潜能 | `caver/sim/traffic.py`, `caver/models/cbc.py` | 0.1s integration + 四分点 swept OBB 近似；碰撞标签、事件发生时停车裕度有效掩码；pair 头预测 risk/clearance/slack；joint 头直接学习 ego/third-party union 风险，后投影保证必要的 pair/union 不等式 | 连续时间保守碰撞证明、真实反应延迟标签、显式训练响应 onset/mode loss、全图 message passing 与可认证的边界近似误差 |
| DSE 联合参考风险集合 \(\mathcal U(D)\) | `caver/validation/evidence.py` | 整场景 max-coordinate 的 split conformal + 有限 K 参考估计补偿；risk LP 保留 \(C_p\le C_{\rm down}\le\sum_p C_p\)；不凭空添加跨世界联合分布 | 无模型偏差覆盖保证；对于不同外部参考驾驶员族、分布漂移和逐场景条件概率，不能直接给出真实道路保证 |
| DSE 风险约束与选择充分性 | `caver/validation/evidence.py` | 区分 robust gate \(\mathcal A_R\)、仅 gated 的 \(\Gamma_R\)、对**任何 pointwise feasible** 候选的 \(\Gamma_P\)；三种状态 `DECISION-CERTIFIED` / `RISK-ADMISSIBLE` / `UNCERTIFIED` | 完整真实候选动态可行集合、真实 short-prefix shield，真实车载执行许可 |
| TIV 时空、对象、证据来源查询 | `caver/validation/tiv.py` | 尝试从 DSE LP witness 提议指定候选、pair、chain、时间段、source 的查询；模拟来源真实计时；学习有符号 surrogate VOI；不允许同一未校准查询直接获得证书 | 按 \(F_\theta\) 对 chain/window 的精确定位、真实 replan/hold 状态一致的查询结果、在线参考模型更新，以及**独立 terminal release-state 校准**和等计算资源的完整闭环对照 |
| WOMD 数据输入 | `caver/data/womd.py` + `caver/data/corridor.py` | Scenario+tf.Example 的官方字段解码、只输出过去+当前、不暴露未来；保存有效历史并变换到 ego 局部系 | v1.3.1 真实 shards 兼容验证、parent 20s segment 去重 manifest、路网拓扑约束与超出 1s 的真实历史补充 |
| nuPlan 输入 | `caver/data/nuplan.py` | read-only DB 清点、官方历史 API、显式时间戳排序、禁止 expert 未来路线输入 | 当前 devkit wheel+DB 在你的服务器验证、将 map API 物化为真实合法道路候选、官方闭环 planner integration |
| ReactSim-Bench | `scripts/download_reactsim.sh`, `scripts/reactsim_preprocess.sh`, `caver/data/reactsim.py` | 官方 PKL/checkpoint 下载指令、软链接和预处理命令、独立输出 schema 检验 | **没有**官方模型闭环运行，不能用 PKL 清点作真实 TIV/RQ3 结果 |
| CARLA 0.9.16 | `caver/integrations/carla_runner.py` | 先导四车同步 runner、碰撞 sensor | 没有与当前四车 synthetic config 同步的合法 spawn/中介 swap/真实动力学一致性检验 |

## 二、这次修复的可导致错误或泄露的关键问题

1. **同一 validation 同时作为 calibration 和 test**。原 `scripts/server_pipeline.sh` 读取同一个 `womd_validation.jsonl`，分别取开头 80 个，导致训练集外的校准结果在测试集复用。现改为官方 `training` 的 `train` 与 `calibration` 互不重叠，官方 `validation` 全部标成 `test`；`audit-splits` 对 ID 和启发式 group 重复强制失败。**仍不可保证**不同地图片段上的 parent 20s 重叠已完全消失。
2. **观察驾驶模型的数据独立性**。修复 `train-policy --womd-train-partition`，强制只使用官方 training 的 `train` 分组，防止 calibration 场景通过观察行为拟合间接进入风险预测网络。
3. **规划时刻历史泄露/丢失**。WOMD 官方 current index 处截断，agent histories 只到本时刻；`corridor_subset` 原丢弃历史，导致 delayed IDM 用现在状态伪装过去状态，已按坐标旋转/镜像修复。历史不足或 invalid 使用掩码，不读取任何未来帧补齐。
4. **TF.Example 结构解析**。按官方 TF.Example schema 显式 `state/id`, `state/type` 为 `float_list`、`state/is_sdc` 为 `int64_list`，独立读取 past/current，`future` 只放在 `include_future` 返回的隔离标签流。真实 v1.3.1 shard 必须进行第 1 份解码的服务器验收。
5. **nuPlan 历史访问和未来 expert route**。只调用当前/过去 API；显式时间排序，禁止 `get_ego_future_trajectory`、`get_expert_goal_state`、`_route_roadblock_ids`。官方 `NuPlanScenario` 没有 `map_name` 公开属性，代码改为读其当前版本 `._map_name` 元数据（待安装版本实测）。
6. **延迟响应器 off-by-one 和事件裕度误标注**。使用匹配的历史 delay 下标；停车裕度改到前车首次减速事件时间并要求纵向几何有效；不再把“无事件”记成安全值 0 或 50m 训练标签。修复后原四车案例结果失效，明确调整搜索后案例输入并记录来源。
7. **第三方 union 风险的数学错误**。不能将不同 pair 的概率相加当 union 碰撞概率。本版 joint head 从一次联合仿真样本直接学 union/ego，后投影只施加必要集合约束。
8. **DSE 只对已过门控候选计算后悔**。已实现单独 \(\Gamma_R\) 和 \(\Gamma_P\) LP，保留那些在某些风险世界可行但不稳健过门控的候选作为比较对象，防止错误声称选择充分。默认 prefix 未验证时不 release。
9. **TIV 查询不是校准证据的任意替换**。查询只在时间、计算预算和已独立确认的安全 hold 前缀约束下可被选择，单次查询不自动授权，已把同一模型追加采样和结构性差异来源分开标识。
10. **碰撞检测字段索引错位（严重）**。原 rollout 的数组布局为 `[x,y,speed,yaw,length,width]`，分离轴碰撞模块却要求 `[x,y,yaw,length,width]`；原实现直接传六列，使 `speed` 被当作航向角、`yaw` 被当作长度，**所有旧几何接触数值须作废**。修订版显式选 `[0,1,3,4,5]`，向量化四轴 SAT，并用随机朝向/尺寸的逐样本标量实现核对。离散四分采样仍非连续时间无碰撞证明。
11. **评估指标“标签”概念混淆**。将参考模拟器的 Monte Carlo 风险频率称为 empirical reference conditional means，**不是** WOMD 实际交通中对未执行 ego 行为的已观测事实；模型对这些均值的平方误差不冒称真实驾驶 Brier 分数。

## 三、你应当首先做的三组证伪实验

### RQ1 / RQ2：CBC 的 event boundary 真的能区分同意图候选吗？

**分两层样本**：`natural cohort` 从事先固定规则的 WOMD/nuPlan 场景池选，不能先看仿真碰撞结果；`enriched challenge cohort` 是按预先公开的 guard proximity/事件敏感性挑选的反例，甚至参数搜索得到的 `fourcar-challenge`，必须单独报告、绝不合并发生率。

在不看实际下游碰撞前先固定 early/late：相同路口/车道目标、终点速度差阈值、进度差 <= 预注册 3m（仅建议初值，需按轨迹动态可行性定义）、加速度/jerk/合法性限制和 8s horizon。只改变时序控制，控制 ADE、候选集合及反应模型预算。**至少**比较 source-only（低保真全候选）、几何距离筛选、CBC 去 guard、CBC 去时延图、CBC 完整版、相同成本的独立全预测。

优先指标：

| 指标 | 解释 / 操作要求 |
|---|---|
| Critical signed contrast sign error | 真实**独立参考模拟器** \(|\Delta C_p|\ge\tau\) 的 pair 中，CBC 对 \(\Delta C_p\) 的符号错判率；报告该有效分母和 95% scene bootstrap CI |
| Critical underresolution | 真 contrast 显著但预测 \(|\widehat\Delta_p|<\tau_{\rm min}\) 的频率，区别于反向错排 |
| Detection AUROC/AUPRC | 按 guard 例外界及真实首次事件预测切换/接触反转；接触极不平衡时重点 PR-AUC |
| Guard-crossing recall @ budget | 在固定 reactive model calls / 实际 wall-clock 内发现的真正边界事件比例；漏检严重度按 pair 记账 |
| Signed clearance MAE and event-valid stop slack | 首次可判定冲突的碰撞裕度、停止裕度只在有效掩码上计算；**同时**报告有效率，防止只在容易样本上算漂亮误差 |
| Onset delay MAE/P90 | 中介 B 首次减速度阈值触发、C 反应延迟，按真实参考源报告 |
| Downstream union risk MAE, reliability/ECE, joint interval coverage | union 由 joint rollout 标注，不能由 pair 直接相加；报告 MC 标签不确定度 |
| Trust-region efficiency vs missed-boundary | 条件 \(\bar\beta\le\eta\) 可复用比例、真实接触反转假阴率、单位场景开销；**在 validated guard bounds 缺失时不能报告可信高效复用率** |
| Fréchet flip/harm bounds | 仅由两候选边际概率得到的跨世界事件上下界；真正个体可归责因果事件不可观测，不能把固定 simulator seed coupling 当成真实群体估计 |

现有 `rq-audit` 已实现 contrast sign/underresolution、pair/union risk error、reference-mean reliability、预测和 source clearance MAE、首次有效事件 slack MAE/有效率。**尚未实现**完整首次制动 onset score、guard 跨越 ROC、真正 validated guard reuse 和地图严格 matched pair；这些是投稿前必须补齐的实验模块，不应标成已经证明有效。

**中介 B->C 核查**：在确定性模型内 exchange B 的 trajectory，观察 C 与碰撞是否交换；对照为断开物理接近的负例、没有 B 制动事件的负例、重新采样的 B、更多不同响应策略。该分析只能验证给定结构模型，不等于从 WOMD 观察数据估计个人因果责任。

### RQ3a：DSE 在不过度拒绝下改善决策吗？

**先冻结** CBC checkpoint、候选集合和参考反应模型族，在开发 calibration 上设 `alpha`、eps_ego/down/inc、regret tolerance 与操作策略；最终 test 只调用一次。`Gamma_R` 基于 robust safe comparator；`Gamma_P` 基于某个风险世界可能安全的 comparator。**两者的差**是“安全可接受、但选择未证充分”的核心检测量。

与 source-risk heuristic、CBC-only 最低风险、只考虑 ego 风险的 planner、只对 \(A_R\) 做 regret 的 DSE（容易虚假决策认证）、完整 \(\Gamma_P\)、允许 risk-only 的预先声明操作策略比较。

同时报告（不允许只挑好的结果）：

- `risk-admissible scene coverage`、`decision-certified scene coverage`、`actual release coverage` 三个不同分母。零覆盖时，`wrong among released` 标成 **N/A**，不是 0%；
- `wrong accepted action rate`（按独立参考风险阈值），总场景错误率，按 released 的下游 union 风险和 ego 增量风险，reference model mixture / 特定独立反应族分层；
- \(\Gamma_R,\Gamma_P,\Gamma_P-\Gamma_R\) 分布，约束 witness 类型、候选 support 缺失原因，`UNCERTIFIED` 与 `RISK-ADMISSIBLE` 各自的触发原因；
- 任务 progress/route compliance/comfort/jerk/持续时间，Pareto 风险—覆盖—进度曲线：**防止只拒绝所有动作而显得安全**；
- 校准场景级同时覆盖率、条件于 release 的经验风险（**普通 split conformal 的边际覆盖率不自动保证被选择/释放条件下的覆盖率**）、校准样本量与源结构 shift 下覆盖率损坏；
- 平均/p95/p99 真实推理和模拟器调用耗时、deadline miss rate、主动 hold 的失效/追尾率。

显式保留 `calibration quantile=1` 时 `UNCERTIFIED` 全部拒绝的案例作为负结果；**不能**通过改 eps 凑 accepted-rate。足够大规模统计时用 scene/log 作为 bootstrap 单位，报告 Wilson/Clopper-Pearson 二项置信区间及分母。

### RQ3b：TIV 的 targeted verification 是否真实降低闭环 decision regret？

训练 acquisition head **仅用 training 的源审计场景**、验证/调参在 development 内完成，保留独立 test 的所有 decision loss 作为 terminal audit；terminal conformal calibration **必须**按冻结的“是否查询/查询哪几个/hold 多久/何时停止/候选集如何变化”的完整策略重新执行，不能把 pre-query quantile 用于 adaptive post-query release。

对照：zero-query、随机查询、最高熵、ensemble disagreement、SIGMA/EPG 型 expected planning gain、均匀分配、所有候选完整高保真推理；**全部同等** p95 耗时/模拟器调用预算和相同 safe-hold 约束。对于新响应族的 TIV，必须独立于 TIV 查询训练模型与终端审计模型；在相同 IDM 结构的不同参数下的“结构独立”只能写 `surrogate cross-regime`，不等于 ReactSim 的独立世界模型。

主要量：

\[
\Delta L_{\rm signed}=L_{\rm exec}(\pi_0)-L_{\rm exec}(\pi_q)-L_{\rm hold}-\lambda_{\rm comp}c_q.
\]

必须允许 **负值**，不能只选择 query 使性能变好的场景。记录 `query source`、链条/窗口/对象、风险 witness、调用次数、回传延迟、超时、执行前缀引起的后车二次风险、post-query 认证覆盖率、终局 regret、安全/效率 Pareto。

当前 `tiv-audit` 只是查询价值**合成代理标签训练**。`verify_source` 不能运行 `physics_envelope` / `fresh_observation`；所谓独立反应族仍共享同一 delayed-IDM 结构。**没有正式 TIV 改善闭环性能的结果**，论文应把该项列为待完成。

## 四、还要补充的负对照与修复检测

1. **Frozen baseline**：固定全部训练超参、参考 reactor family 和评估候选后才看 final test；不根据 test 去搜索碰撞反例，也不把 posthoc challenge 混入自然集。
2. **No future**：对同一 WOMD 场景，强行把未来 80 帧替换成无穷远随机位置，`read(...with_future=False)` 和模型当前规划输入必须逐字节/数值相同；当前已有 TF.Example mock 版单测，真实 shard 版待服务器运行。
3. **Valid mask**：对历史中无效检测不进行坐标/速度插值伪造；只用合法时刻历史；当 C 在 t0 尚未进入观测时，不允许从其未来轨迹“提前生成”车辆。
4. **nuPlan complete-log split**：同一 DB 不得同时出现在 train/cal/test；同位置相邻时刻、多个 scenario token 不得当独立 episode。
5. **WOMD overlap manifest**：从 `training_20s` 或官方 segment ID 提供 parent ID 映射；没有准确映射时，模型训练/校准独立性只能标为启发式，不能声称严格可证明。
6. **Paired counterfactual**：同一模拟反应潜变量用于比较两个候选可降低 Monte Carlo 方差，但不能被宣称为已从真实世界识别的 cross-world potential outcome；给出 Fréchet 界。
7. **Traffic law**：当前 corridor filter 是纯几何，不是地图车道拓扑/红灯合规约束；论文正式样本必须按当时红灯、通行权、路线、车辆动力学规则重新过滤。
8. **CARLA bias check**：与合成 IDM 反应族交叉验证，独立控制器如 CARLA traffic manager 与官方路口规则；报告未复现机制的场景。
9. **OOD / structure shift**：驾驶员延迟、急刹模式、天气、地区、不同地图场景、仿真架构独立切分，报告 risk–coverage 曲线与 failure category。
10. **Published uncertainty**：校准只对可交换参考分布的全场景风险矢量提供统计主张，单场景的真实道路条件安全以及查询后 release 需要新的独立保证。

## 五、通过测试和合成结果的解释

本版在构建环境中运行了 **40 项自动测试**（全部通过，包含无未来 TF.Example 模拟、延迟历史、首次事件有效性、DSE `Gamma_P`、split 重叠保护、TIV 门控与 CBC 联合风险输出）。旧四车案例参数在延迟索引修复后被标记为需要重新调参；当前 `configs/fourcar_challenge.json` 是已明示后验修订的合成固定案例，而**不是** CARLA 或自然交通证据。

小规模合成 smoke 使用 5 train、6 calibration、4 test 场景，每场景 2 source worlds 和每审计族 3 reference worlds。本次在修复几何后运行的 smoke **截至独立校准阶段已完成**，得到 `quantile=1.0`、`mc_buffer≈0.826`、`n=6`；完整脚本在本地 120s 单次执行限制下未执行到最终阶段，因此**不宣称整条最新流程通过**。完整测试及评估应在服务器重新执行。不过，随后已用 **2 个测试场景与 1 个模型训练 epoch** 完成小预算后半链路（`audit-splits`、`rq-audit`、`evaluate`、`tiv-audit`），结果为 `critical_pair_tests=0` 和 2/2 `UNCERTIFIED`；只说明命令与数据接口可执行，不构成效果验证。这不是 DSE 性能成功，应报告为“未证实可用覆盖率”。此处的样本极小，**不**作性能估计、p 值、论文安全结论。

## 六、论文投稿前的最低完成标准

- **CBC**：实现完整候选条件时延 GNN/状态递推，并给出在明确完整 guard 与物理扰动包络下有记录可检验的 `beta_bd`，运行独立响应模型、延迟/事件标签、可失败的 matched pair 安全辨别对照；报告计算时间。
- **DSE**：在官方独立数据/反应闭环环境中验证联合参考校准与真实物理 prefix shield，报告非零覆盖与对应错误接受风险/后悔，不能只展示全拒绝。
- **TIV**：实现正式 ego 候选执行、主动 hold 导致新的当前观测与状态的后查询校准，完成至少一种完全独立响应源的等时间预算对照。
- **数据**：实测 WOMD v1.3.1 Scenario 解码/父片段去重；nuPlan 官方 DB+地图接口；ReactSim checkpoint+真实 rollout；CARLA 0.9.16 在合法地图上可复现/否证同意图级联机制。
- **统计**：自然分布及经搜索/增强的 challenge 分开，场景或 log 级 confidence interval，全测试中没有选取最好的 seed/posthoc 阈值。

**结论**：本版是一套更严谨的**可执行研究基线与证伪审计框架**，不是已经完全对齐论文全部公式的可投稿 SOTA 结果。最关键剩余阻塞是 CBC 完整时延事件边界传播网络、经过验证的增量误差包络、独立 reactive closed-loop 数据及 TIV 终端 release-state 校准。

### 修复几何后的冻结四车数值（仅后验合成案例）

| 合成候选 | Ego 接触 | B–C 接触 | B–C SAT 分离轴裕度 | Ego–B SAT 分离轴裕度 |
|---|---|---|---:|---:|
| Early | 无 | 有 | −1.800 m | +2.859 m |
| Late | 无 | 无 | +1.509 m | +5.601 m |

交换 B 的强制轨迹会交换 B–C 接触状态；两个 Ego 候选最大位置偏差 2.772m。SAT signed gap 不是正交欧氏间距，也不是穿透深度。旧报告 `−1.249/+0.580` 和修复前其他任何几何数字都不再有效。
