# recipe-lab

Closed-loop experiments for semantic-router routing recipes（RSI 工作台）。

本层覆盖 replay-tier 闭环：冻结题集身份（A4）→ 题集三分（B）→ 答案库与归因 →
候选提议（信号级 / 规则级）→ replay 层评估入台账（A1/A5）→ 台账检索与洞察引用（G1/G2）。
live 层随 Sub-issue 3 到来。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## 闭环工作流

1. **题集三分（B）**：把带标签题集按组确定性切成 calib / dev / frozen_test。
   frozen_test 隔离：replay / propose / attribute 一律拒绝触碰，只留给 live-verdict --final。
2. **冻结身份（A4）**：spec 的 workload 记录每个 split 的 `content_hash`
   （`sha256:` + 原始字节哈希）；attribute / run / propose-thresholds 在触数据前
   重算校验，任何漂移直接拒绝（exit 2）。
3. **答案库（bank）**：绕过 router 直接问每个模型，按 `expected_contains` 评分。
4. **归因（attribute）**：E1 信号错 / E2 策略错 / E3 能力错（+ 次优路由、无证据）。
5. **诊断（D）**：`diagnose` 对无标签 replay 流量做三类体检——决策覆盖率（dead/rare，
   `--expect-rare` 里的安全决策只报告不告警）、信号分布（分位点 / 自身量程直方图 /
   量程利用率）、阈值可达性（规则谓词位点按写入语义求流量满足率，全组为 0 即 unreachable）。
   产出 `diagnosis.json`（`recipe-lab-diagnosis-v1`，带输入哈希，字节级稳定）与 `diagnosis.md`；
   顶层 `dead_decisions` / `rare_decisions` / `unreachable_sites` 数组与 `next --diagnosis`
   的消费形状一致。诊断只报告现象，不是裁决。
6. **候选提议**：`propose-thresholds` 扰动单个决策信号阈值；
   `propose-rule-thresholds` 一次性移动一个信号在决策规则里的全部谓词位点
   （含 NOT 分支；BUG-3 黄金用例即四个 `difficulty gte 2.5 -> 1.75` 一次到位）；
   `propose-calibrated` 从诊断产物的观测分布取分位点（或目标流量占比）自动定值，
   旁车自动携带 `diag:...#threshold.<decision>.<signal>` 引用，诊断与 base config
   哈希不符直接拒绝（exit 2）。
   每个候选除 YAML 外还写 `<stem>.candidate.json` 旁车
   （kind / ref / change_summary / provenance / sites）。
7. **replay 评估（run）**：同一题集 + 答案库下重放候选配置。排序指标是契约里的
   `objective.replay_metric`（A1，默认 `routed_accuracy`）；`budget.rounds` 是轮次硬上限，
   `budget.max_minutes_per_trial` 是单个 trial 的 deadline（A5），超时记 `status=timeout`
   且不参与排序。`--validate` 用 `vllm-sr config validate` 预检候选，被拒候选不进台账，
   全部被拒则 exit 2。
8. **台账（ledger）**：append / show / top。每条记录有稳定 `record_id`（G1），
   provenance 可携带 insight-refs（G2）。`ledger top --spec` 按 replay_metric 取最优。

## CLI

全部子命令：

| 命令 | 作用 |
|---|---|
| `plan` | 校验并展开实验契约 |
| `hash` | 计算题集冻结身份：sha256 原始字节（A4） |
| `split` | 组级确定性三分 calib/dev/frozen_test（B） |
| `attribute` | E1/E2/E3 误路由归因（`--split` + 可选 `--cases` 哈希校验） |
| `collect-replays` | 从管理 API 拉取 router_replay 记录并连接标签 |
| `diagnose` | 无标签流量的三类体检：覆盖率 / 分布 / 阈值可达性（D） |
| `bank` | 构建 / 应用 per-case/per-model 答案库 |
| `propose-thresholds` | 扰动单个决策信号阈值 |
| `propose-rule-thresholds` | 移动一个信号的所有规则级谓词位点（C） |
| `propose-calibrated` | 按观测分布分位点提议规则阈值候选（D） |
| `run` | replay 层评估候选 → 台账 RANKED（A1/A5/C/G2） |
| `ledger` | 台账 append / show / top（G1/G2） |

常用示例：

```bash
# 校验并预览实验契约（不执行任何实验）
recipe-lab plan --spec examples/spec.example.yaml

# 三分题集（同题集 + 同参数 => 字节级一致输出）
recipe-lab split --cases labeled.jsonl --output-dir splits/

# 诊断：不需要标准答案，从无标签 replay 流量里发现死决策、压缩量程与不可达阈值
recipe-lab diagnose --records rows.jsonl --config config.yaml --output-dir diag/ \
  --expect-rare local_security_containment,local_privacy_policy

# 候选生成：BUG-3 式校准（规则级位点一次到位）
recipe-lab propose-rule-thresholds --base-config config.yaml \
  --signal difficulty --from 2.5 --values 1.75 --output-dir cands/

# 候选生成：按诊断观测分布自动定值（p75，或 --target-share 0.25 表达同一意图）
recipe-lab propose-calibrated --diagnosis diag/diagnosis.json --base-config config.yaml \
  --signal difficulty --from 2.5 --quantiles 0.75 --output-dir cands/

# replay 评估（frozen_test 拒绝；候选先过 config validate）
recipe-lab run --spec spec.yaml --split dev --cases splits/dev.jsonl \
  --bank bank.jsonl --candidates cands/ --ledger ledger.jsonl \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py \
  --output-dir out/ --validate

# 台账
recipe-lab ledger show --file ledger.jsonl --tier replay --json
recipe-lab ledger top --file ledger.jsonl --spec spec.yaml
```

## 洞察引用（--insight-refs，G2）

`run` 与 `ledger append` 接受逗号分隔的 provenance 引用，合并进记录的
`candidate.provenance.insight_refs`（去重）；不可验证的引用直接拒绝（exit 2）：

- `rec-<id>`：台账里已存在的 record_id
- `bench:<run_id>`：基准 run id（仅格式校验）
- `attr:<path>@sha256:<hex>`：归因产物，哈希必须与文件字节一致
- `diag:<path>@sha256:<hex>#<anchor>`：诊断产物，锚定到文件内某个位置

## 记录格式（归因输入）

每行一个 JSON 对象（router_replay 适配器产出同此形状）：

```json
{"case_id": "c1", "truth": "math", "matched_decision": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null`（null = 无证据，按证据优先原则不计为错误）
- `family`: 产生该判定的信号家族（`vela_classify` / `vela_decisions` / `d3`），归因按家族分列

## 契约要点

- `objective.replay_metric` 是 replay 层唯一排序指标（A1）；旧 `evaluation` 段已退役，
  加载即报错（A5）。
- `workload` 声明三分 split（B）时必须给出 `split_rule` 文本且三分互不相交；
  旧的单数据集形态按 dev-only 处理并告警。
- `budget.max_minutes_per_trial` 覆盖就绪等待 + 请求循环 + replay 拉取全程（A5）。

## 测试

```bash
pytest
```

## Smoke 脚本（examples/smoke/）

smoke 脚本全部从环境变量取环境，未设置时按脚本自身位置推导合理默认值，clone 后即可在任意 Linux 机器运行：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `REPO` | semantic-router checkout | 由脚本位置推导 |
| `RECIPE_LAB_BIN` | recipe-lab CLI 与 python 所在目录 | `$REPO/.venv/bin` |
| `PYTHON_BIN` | mock backend 使用的 python | `$RECIPE_LAB_BIN/python` |
| `SR_BUILD` | Router 构建前缀 | `$REPO/build` |
| `ROUTER_BIN` | Router 二进制 | `$SR_BUILD/bin/semantic-router` |
| `WORK_ROOT` | 运行产物根目录 | `$REPO/.tmp-smoke-runs` |
| `DATASET_DIR` | 本地 MMLU-Pro 快照（live/e2e 用） | `$WORK_ROOT/dataset/mmlu-pro` |
| `HF_HOME` | HuggingFace 缓存 | `$WORK_ROOT/hf-home` |

MMLU-Pro 快照用 huggingface_hub 获取：

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('TIGER-Lab/MMLU-Pro', repo_type='dataset', local_dir='$DATASET_DIR')"
```

需要 GPU 的脚本（Router 加载决策模型）在 SLURM 集群上提交单个任务即可，例如：

```bash
srun --gres=gpu:1 --cpus-per-task=8 --time=00:45:00 \
  bash tools/recipe-lab/examples/smoke/replay-smoke.sh
```

`calibration-smoke.sh` 是唯一的例外：纯离线（无 GPU、无 router），用合成夹具完整走一遍
`diagnose` → `propose-calibrated`，并断言候选与 BUG-3 黄金结果一致；同一流程也写成
pytest 用例（`tests/test_calibration_smoke.py`）纳入单元测试。

集群级运维脚本（数据集批量下载、长时间循环压测、作业收尾）不属于本仓库，放在仓库外私有目录维护。
