> **2026-11-05 CBC-first v2 提醒：本文件为上一版历史记录；最新版运行/评估以 [`CBC_FIRST_RUN_CN.md`](CBC_FIRST_RUN_CN.md) 与 [`CBC_V2_CODE_AUDIT_CN.md`](CBC_V2_CODE_AUDIT_CN.md) 为准。新的 `scripts/server_pipeline.sh` 是限额压缩 WOMD 的 CBC-first 脚本，不再导出无界的 `womd_training.jsonl`，也不联合宣布 DSE/TIV 已认证。**

# CaVER 修订版：完整执行手册（2026-11-04）

> **适用范围**：这是对上传的 `interactive planning.tex` 与旧版 `CaVER.zip` 逐项核查后的**研究原型修订版**。现有代码提供可运行的合成机制、WOMD/nuPlan 观测读入、低保真候选反事实采样、CBC 风险预测、DSE 联合风险/选择判定与 TIV 离线查询审计。**未提供经过官方闭环验证的 CBC 全模型、Waymax/ReactSim 反应式驾驶规划器或真实道路安全证书。** 因缺乏你服务器的 `/data0` 数据，真实数据指令为待执行的预检/试验，**不可声称已跑通**。从项目 `CaVER/` 根目录运行全部命令。

## 1. 数据集与分工——只下载实际需要的内容

**WOMD 官方 v1.3.1**：

```text
/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/
└── uncompressed/
    ├── scenario/
    │   ├── training/               [本版必需：历史训练和训练池的校准子集]
    │   ├── validation/             [本版必需：内部最终保留测试]
    │   └── training_20s/           [第二阶段：分组、长时间迟滞事件]
    └── tf_example/
        ├── training/               [可选：Waymax 原生读入]
        └── validation/             [可选：Waymax 的官方基线]
```

- `scenario` 与 `tf_example` **不是两套独立样本**。本项目只用 `scenario` 训练 CBC，`tf_example` 仅在选择 Waymax 基线时单独读取，不能合并。WOMD 官方九秒窗口通常为 10 帧过去、1 帧当前、80 帧未来（10Hz），真实 planning instant **只使用过去与当前（共 11 帧）**，未来 80 帧只供隔离的离线标注研究，不能输入规划、风险回归或 TIV。
- `scenario/training_20s` 不应混入标准 9 秒训练数；它用于第二阶段真实源片段去重、较长历史/制动统计。在具备官方父 segment/log 映射前，**按地图 ID 分组仅为启发式防泄露，无法证明相邻 9s 窗口绝对独立**。
- `validation_interactive` 是专用交互预测评估集合，**不是**本论文评估闭环规划效果的必选训练集合。测试集的未来标签不作为本地监督。
- v1.3.1 的 `sdc_paths` 可包含候选有效路径；**不得把未来已实现路线、参考专家未来轨迹或人工定义的行为剧本当成当前可观测未来**。本适配器暂不使用此字段。

**nuPlan**：保持你已经下载的 DB 数据不变：

```text
/data0/senzeyu2/dataset/nuplan/data/cache/
├── public_set_boston_train/
├── public_set_pitts_train/
├── public_set_sg_train/
├── public_set_vegas_train/
├── public_set_val/
└── public_set_test/
/data0/senzeyu2/dataset/nuplan/maps/
```

`public_set_*_train` 用于训练；`public_set_val` 先按完整 DB/log 组拆为开发校准和评估，不跨文件泄露；`public_set_test` 暂时封存。当前 `nuplan-export` 通过官方 devkit 只读取当前及过去的 ego、车辆轨迹；明确**不调用可能从未来 expert 轨迹推导路线的 API**，也不把这些 DB 样本直接宣称为闭环仿真。基于独立地图和当前导航任务构建合法候选的功能仍待实现。

**ReactSim-Bench**：对应目录为：

```text
/data0/senzeyu2/dataset/ReactSim-Bench/
├── repo/                 [官方 GitHub 源码]
├── ckpts/                [官方模型 checkpoint]
└── data/
    ├── custom_behaviors/ [官方行为测试 PKL]
    └── nuplan/
        ├── maps -> /data0/senzeyu2/dataset/nuplan/maps/
        └── nuplan-v1.1/splits/trainval/ -> [各个 DB 的软链接]
```

行为 PKL、nuPlan DB、预处理缓存、motion anchors、模型权重和官方 rollout runner **全部参与**最终 ReactSim 实验，不能把清点 PKL 的命令称作闭环评估。

## 2. 安装 CaVER 的核心环境

建议使用 Python 3.10 或 3.11 的**独立核心环境**，不要把老 nuPlan、CARLA、TensorFlow 与 PyTorch 的相冲突依赖装进一个环境：

```bash
unzip CaVER_revised_2026-11-04.zip
cd CaVER
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
python -m pytest -q
bash scripts/smoke.sh
```

如果在 Python 3.11 环境，则把 `python3.10` 换成已存在的 `python3.11`。`core` 不依赖 TensorFlow/nuPlan/CARLA；测试采用 mock/合成数据。

### 2.1 WOMD protobuf 环境（需在服务器决定兼容版本）

推荐 Python 3.10 独立环境，安装相互匹配的 TensorFlow 和官方 Waymo protobuf。以下是**一个需验证 pip wheel 可用性的示例**，不能保证在你服务器的 CUDA/Python 环境中即刻安装成功：

```bash
# 在单独 conda/venv 环境，而非 CARLA 环境中执行：
python -m pip install 'tensorflow==2.12.*' 'waymo-open-dataset-tf-2-12-0'
python -c 'import tensorflow as tf; from waymo_open_dataset.protos import scenario_pb2; print(tf.__version__,scenario_pb2.Scenario.__name__)'
# 该环境仍需安装上面的 CaVER 核心包（pip install -e .）。
```

TF 版本如不可用，应从官方 Waymo Open Dataset 的 wheel 与现有 CUDA/Python 矩阵选择兼容组合，而不是盲目卸载现有环境。`tf_example` 原生 Waymax 需要另装 `waymax` 与匹配 JAX/TF：

```bash
python -m pip install waymax
python -c 'import waymax; print("Waymax import OK")'
```

**官方下载**：先在 Waymo 官网同意相应许可协议，然后根据官方 v1.3.1 Google Cloud Storage 目录只同步需要的两组：

```bash
export WOMD=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
mkdir -p "$WOMD/uncompressed/scenario"
# 先使用 gsutil ls 或 gcloud storage ls 确认已授权且 bucket 可访问：
gsutil ls 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/'
# 示例：仅同步需要的官方文件（在已取得访问权限时）：
gsutil -m rsync -r 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/' "$WOMD/uncompressed/scenario/training/"
gsutil -m rsync -r 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/' "$WOMD/uncompressed/scenario/validation/"
# 如果要运行 Waymax 的原生 TFRecord 基线：
mkdir -p "$WOMD/uncompressed/tf_example"
gsutil -m rsync -r 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/' "$WOMD/uncompressed/tf_example/training/"
gsutil -m rsync -r 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/' "$WOMD/uncompressed/tf_example/validation/"
```

请先 `gsutil ls` 核实远端路径、许可及配额再复制；官方工具有时使用 `gs:///` 形式的公开地址，真实 bucket 是否可以匿名访问取决于账户/许可。不下载 `lidar_and_camera`，本算法现阶段不使用其额外输入。

## 3. 离线合成实验（无需真实数据库）

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python -m caver.cli fourcar-challenge --output outputs/fourcar_challenge.json
python -m caver.cli fourcar-mediation --out outputs/mediator_audit.json
bash scripts/smoke.sh
```

`fourcar-challenge` 是**合成搜索后又重新调整过的冻结案例**，因为修复感知延迟下标后旧四车参数不再产生原预期接触。不能把搜索后案例当成从 WOMD/nuPlan 随机采样到的真实发生频率，不能用它替代 CARLA 或 ReactSim 的验证。`smoke` 为完整合成流水线，需要根据 CPU/系统负载允许足够的执行时间；本地修订后的最新完整流程**尚未全部通过**，目前已完成到 calibration。脚本生成 train/calibration/test 三组**不交叉的合成随机种子**，运行 CBC、联合校准、DSE 与 TIV 离线代理指标。

## 4. WOMD 服务器数据预检：**先全部通过，后训练**

```bash
bash scripts/verify_womd.sh
python -m caver.cli nuplan-check --output outputs/nuplan_inventory.json
python -m caver.cli reactsim-check --output outputs/reactsim_inventory.json
# 只有已下载 tf_example 并配置 Waymax 后才运行可选检查：
CHECK_WAYMAX=1 bash scripts/verify_womd.sh
```

或单独执行：

```bash
python -m caver.cli doctor --output outputs/doctor.json
python -m caver.cli womd-check --format scenario --split training --decode 3 --output outputs/womd_training_check.json
python -m caver.cli womd-check --format scenario --split validation --decode 3 --output outputs/womd_validation_check.json
python -m caver.cli womd-check --format tf_example --split training --decode 3 --output outputs/womd_tf_training_check.json
python -m caver.cli waymax-read --split training --limit 2 --output outputs/waymax_scenes.jsonl
```

检查 `decoded_checks[].agents`, `ego`, `future_labels`。注意 `womd-check` 为了审计读取了独立的 `with_future=True` 标签流；**planning Scene 对象绝不含 future**；正常 `womd-export` 不导出 future。遇到 schema shape、`state/id` 类型、TFRecord 压缩、map feature 等不一致，**停止训练并保存完整错误日志**。

## 5. 训练、校准、测试：严格执行顺序

**先跑少量 pilot**，确认预计 100/80/80 个符合四车纵向结构的场景可以从当前 30k/12k 样本中采到；如果报 `Zero eligible ...`, 先检查几何筛选，不得以读测试未来轨迹来补样本。完整指令在已修订的脚本中：

```bash
# 本脚本不需要 tf_example / Waymax，默认仅使用 scenario。
# 你也可以将它们设置成更小的 pilot 预算：
EXPORT_TRAIN=30000 EXPORT_VALIDATION=12000 \
TRAIN_SCENES=100 CAL_SCENES=80 TEST_SCENES=80 \
SOURCE_WORLDS=8 AUDIT_WORLDS=64 CBC_EPOCHS=80 \
bash scripts/server_pipeline.sh 2>&1 | tee outputs/server_pipeline.log
```

该脚本按以下次序运行（**不可颠倒**）：

1. 预检 Scenario `training`、`validation` 和依赖；
2. 分别导出**当前+历史**场景，未来标签禁止进入输入；
3. 只在官方 `training` 的 `train` 分组学习已观测驾驶员策略（`train-policy --womd-train-partition`）；
4. 使用 `training` 的 `train` 分组构建反事实训练轨迹并拟合 CBC；
5. 使用**同一个官方 training 语料中未参与拟合的 `calibration` 分组**，构建独立参考模拟器轨迹并获取 DSE 阈值；
6. 使用官方 `validation` 全部作为 held-out 内部测试；
7. `audit-splits` 拒绝场景 ID / 启发式地图组的交叉；
8. 冻结 checkpoint，在校准集拟合一次全场景概率区间；
9. 在测试集运行 CBC 分辨率审计、DSE 风险与选择判定；
10. TIV 查询价值拟合**只用 train**，没有后查询安全证书。

可手动重跑独立审计：

```bash
python -m caver.cli audit-splits --train outputs/train.jsonl --calibration outputs/calibration.jsonl --test outputs/test.jsonl --output outputs/split_audit.json
python -m caver.cli rq-audit --input outputs/test.jsonl --checkpoint outputs/cbc.pt --output outputs/rq_audit.json
python -m caver.cli evaluate --input outputs/test.jsonl --checkpoint outputs/cbc.pt --calibration outputs/calibration.json --output outputs/evaluation.json
```

`evaluate` **默认不释放任何实际控制动作**，因为没有经过验证的真实执行前缀。只为合成**数值覆盖率的上界/敏感性试验**可以加：

```bash
python -m caver.cli evaluate --input outputs/test.jsonl --checkpoint outputs/cbc.pt --calibration outputs/calibration.json --surrogate-prefix-check --output outputs/experimental_SURROGATE_PREFIX_ONLY.json
```

不要把标志 `--surrogate-prefix-check` 解释为物理真实安全检测。达到 deadline 后需要真实的、独立验证的短前缀 shield；**绝不能通过将 `strict_prefix=True` 伪造实际控制安全性**。

如果 calibration 的 `quantile=1`、全部 `UNCERTIFIED`，这是样本量小或模拟分布不匹配的合理阴性结果。不能为了得到正例而看测试集调 `eps_down`、`eps_reg`、alpha 或在 test 上重新拟合 CBC。

## 6. ReactSim-Bench 官方下载、预处理及运行

官方：<https://github.com/Thinklab-SJTU/ReactSim-Bench>、<https://huggingface.co/datasets/zzzy0729/ReactSim-Bench-Data>、<https://huggingface.co/zzzy0729/ReactSim-Bench-Models>。

```bash
# 从 CaVER/ 根目录执行；会安装 hf CLI 并下载 PKL/checkpoint，克隆官方 repo，创建 DB/地图软链接
bash scripts/download_reactsim.sh
# 单独创建 Python 3.9 环境，避免与 TensorFlow/CaVER/CARLA 依赖冲突：
conda create -n reactsim_dp python=3.9 pip -y
conda activate reactsim_dp
bash scripts/reactsim_preprocess.sh 2>&1 | tee /data0/senzeyu2/dataset/ReactSim-Bench/preprocess.log
```

这一步仅运行**官方数据预处理**，不是官方闭环评估。官方 `reactsim_data_pipeline/preprocess.sh` 当前将缓存写入 `repo/data/test/nuplan_processed/{train,val}`，本版脚本创建不覆盖文件的兼容软链接 `repo/data/nuplan_processed`。你现有 `public_set_*` DB 不一定包含所有 2,636 个 ReactSim 测试场景的原始日志；若官方 scenario ID 找不到对应 DB，必须补下缺失 nuPlan 原始 DB，不得用未来轨迹回填。

随后按官方仓库文档分别安装 SMART/CATK/CTG 等所选反应模型环境，并执行**有 ego 控制输入的官方 reactive closed-loop runner**。本项目**没有自动拼接一个假的官方模型入口**。只有在得到完整真实多车 rollout 后，按 `docs/REACTSIM_EXPORT_SCHEMA.md` 导出机器可查的 JSONL，然后：

```bash
# 可运行的只是结构/来源字段校验，而非真实物理重新仿真：
python -m caver.cli reactsim-check --output outputs/reactsim_inventory.json
python -m caver.cli reactsim-check --export outputs/official_reactsim_rollouts.jsonl --output outputs/reactsim_export_audit.json
```

## 7. nuPlan 当前+历史预检与导出

```bash
python -m caver.cli nuplan-check --output outputs/nuplan_inventory.json
DB=$(find /data0/senzeyu2/dataset/nuplan/data/cache/public_set_val -name '*.db' | head -n 1)
[[ -n "$DB" ]] || { echo 'No public_set_val DB found'; exit 1; }
python -m caver.cli nuplan-check --db "$DB" --output outputs/nuplan_schema.json
# 在装有官方 nuplan-devkit 的独立环境中执行：
python -m caver.cli nuplan-export --db "$DB" --map-root /data0/senzeyu2/dataset/nuplan/maps/ --limit 10 --output outputs/nuplan_scenes.jsonl
```

注意，`nuplan-export` 使用官方 `get_ego_past_trajectory`、`get_past_tracked_objects`、`get_past_timestamps` 和当前帧；当历史样本数量、时间戳、地图资源不一致时将报错而非默默复用未来。真实 `NuPlanScenario` API 版本依赖必须现场验证。此命令输出的静态 map **仅有 map API 元数据，尚无经过地图道路拓扑认证的候选集合**。

## 8. CARLA 0.9.16

你已安装 CARLA 0.9.16。开一个终端启动服务器（按实际安装目录设变量）：

```bash
export CARLA_HOME=/path/to/CARLA_0.9.16
"$CARLA_HOME/CarlaUE4.sh" -RenderOffScreen -carla-port=2000
```

另一个终端，先确认 PythonAPI `import carla` 与 0.9.16 服务端相容，然后从项目根目录执行：

```bash
python -c 'import carla; print("CARLA Python API import OK")'
python -m caver.cli carla-run --town Town04 --out outputs/carla --step 0.05 --horizon 8.0
```

**CARLA runner 是探索性四车物理先导实验**：目前不能自动把 `configs/fourcar_challenge.json` 逐坐标映射成 Town04 的合法车道 + 同一动力学，亦不能证明反事实车辆分支可比。执行结果需要审查初态、碰撞传感器、道路几何、相同反应种子及链式 B--C 中介交换，才可支持论文结论。

## 9. 推荐回传的首批文件

只需提交服务器输出中的 `outputs/doctor.json`、`outputs/womd_training_check.json`、`outputs/womd_validation_check.json`、`outputs/nuplan_inventory.json`、`outputs/split_audit.json`、`outputs/calibration.json`、`outputs/rq_audit.json`、`outputs/evaluation.json` 和报错日志（如果失败）。它们可定位是数据 schema、划分、置信区间还是闭环响应模型出了问题。

## 10. 论文引用与来源

- 最新论文：本项目随交付更新时以用户提供的 `interactive planning.tex` 为主，特别是 CBC 的 `guard/exceptional mass`、DSE `A_R` vs `Gamma_P` 和 TIV 终端 release calibration 约束。
- WOMD 官方：<https://waymo.com/open/data/motion/>；版本下载页 <https://waymo.com/open/download/>。
- Waymax 官方 config：<https://github.com/waymo-research/waymax/blob/main/waymax/config.py>。
- nuPlan devkit：<https://github.com/motional/nuplan-devkit>。
- ReactSim-Bench：<https://github.com/Thinklab-SJTU/ReactSim-Bench>，预处理说明 <https://github.com/Thinklab-SJTU/ReactSim-Bench/blob/main/docs/install_and_preprocess.md>。
