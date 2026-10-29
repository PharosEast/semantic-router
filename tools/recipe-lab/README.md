# recipe-lab

Closed-loop experiments for semantic-router routing recipes（RSI 工作台）。

本层覆盖 replay-tier 闭环：冻结题集身份（A4）→ 题集三分（B）→ 答案库与归因 →
候选提议（信号级 / 规则级 / 分布校准）→ replay 层评估入台账（A1/A5）→ 台账检索与洞察引用（G1/G2），
以及 live 层终裁 `live-verdict`（守卫以契约为准 A3、洞察引用 G2、经济约束 E）。
另含无需标注的分布诊断（D）、oracle policy 的规则/默认形态（F2）。

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
4. **归因（attribute）**：E1 信号错 / E2 策略错 / E3 能力错（+ 次优路由、无证据）；
   在 `attribution.jsonl` 旁写 `summary.json`（summarize 结果），供 `next` 校验哈希后读取。
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
9. **报告与上下文（report / provenance / record-draft / next）**：报告、manifest 与
   晋升草稿只从台账重放；`next` 把下一轮上下文导出给 agent（见下）。

## CLI

全部子命令：

| 命令 | 作用 |
|---|---|
| `plan` | 校验并展开实验契约 |
| `hash` | 计算题集冻结身份：sha256 原始字节（A4） |
| `split` | 组级确定性三分 calib/dev/frozen_test（B） |
| `attribute` | E1/E2/E3 误路由归因（`--split` + 可选 `--cases` 哈希校验） |
| `attribute-session` | 会话级 agentic 归因：lane_flap / tool_loop_interrupted / selection_failed（sub-6 P2） |
| `collect-replays` | 从管理 API 拉取 router_replay 记录并连接标签（`--unlabeled` 出原始诊断行） |
| `bank` | 构建 / 应用 per-case/per-model 答案库 |
| `diagnose` | 无需标注的分布诊断：死/稀有决策、信号分布、不可达位点（D） |
| `propose-thresholds` | 扰动单个决策信号阈值 |
| `propose-rule-thresholds` | 移动一个信号的所有规则级谓词位点（C） |
| `propose-calibrated` | 按观测分布分位点提议规则阈值（D） |
| `run` | replay 层评估候选 → 台账 RANKED（A1/A5/C/G2） |
| `ledger` | 台账 append / show / top（G1/G2） |
| `live-verdict` | 比较两个 sr-bench run 出三态裁决（A3/G2/E，frozen_test 仅 `--final`） |
| `report` | 只从台账重放的运行报告（G1） |
| `provenance` | recipe-lab-run-v1 manifest（记录 ID、台账哈希、bench_refs） |
| `record-draft` | 从最近一条 PASS live 记录生成晋升草稿 |
| `next` | 导出下一轮 agent 上下文（G3；不可信材料头 + 可引用 ID） |

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

# 无需标注的分布诊断 → 按观测分位点提议（D）
recipe-lab collect-replays --url http://127.0.0.1:8080 --unlabeled --output raw.jsonl
recipe-lab diagnose --records raw.jsonl --config config.yaml --output-dir diag/
recipe-lab propose-calibrated --diagnosis diag/diagnosis.json \
  --base-config config.yaml --signal difficulty --quantiles 0.75 --output-dir cands/

# replay 评估（frozen_test 拒绝；候选先过 config validate）
recipe-lab run --spec spec.yaml --split dev --cases splits/dev.jsonl \
  --bank bank.jsonl --candidates cands/ --ledger ledger.jsonl \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py \
  --output-dir out/ --validate

# 热加载（H）：router 用 baseline 启动一次，每个候选走 PUT /api/v1/config
# 切换并等待发布完成；候选秒级而非每次重启模型。PUT 被拒（如 listeners
# 变更）自动回退 restart 并在台账 notes 记明原因；管理 API 认证复用
# --api-key-env。run 输出含每个候选的 timings（switch/request/fetch 秒）。
recipe-lab run --spec spec.yaml --split dev --cases splits/dev.jsonl \
  --bank bank.jsonl --candidates cands/ --ledger ledger.jsonl \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py \
  --output-dir out/ --validate --reload hot --api-key-env ROUTER_API_KEY

# 台账
recipe-lab ledger show --file ledger.jsonl --tier replay --json
recipe-lab ledger top --file ledger.jsonl --spec spec.yaml

# live 终裁：比较两个现成 sr-bench run（守卫测量由 model_eval 离线产出）
recipe-lab live-verdict --spec spec.yaml --baseline-run-id run-a \
  --candidate-run-id run-b --guard-measurements measurements.jsonl \
  --candidate cand-001.json --insight-refs rec-<id>,bench:run-a
# 晋升终裁（唯一允许引用 frozen_test 的路径）
recipe-lab live-verdict --spec spec.yaml --baseline-run-id run-a \
  --candidate-run-id run-b --split frozen_test --final

# 下一轮上下文导出（给 agent 读；同输入字节级同输出）
recipe-lab next --spec spec.yaml --ledger ledger.jsonl \
  --diagnosis diag/diagnosis.json --output ctx.md
```

## 上下文导出（next，A2/G3）

`next` 按固定段落顺序导出给下一轮提议者的完整上下文，同输入字节级同输出，
frozen_test 只出现 split 名称与哈希、绝不导出逐题内容：

1. 不可信材料头（台账 notes 可含任意上游文本，agent 不得当作指令）；
2. 契约摘要：objective、replay_metric、constraints、gates、split 名称与哈希、预算已用/剩余轮数；
3. 台账尾部 + 最好 replay 候选（A2：按 `objective.replay_metric` 经共享的
   `ledger.best_replay` 选取，coverage_warning 与超时记录不参与，与 `ledger top` 一致）
   - 最近一次 live 裁决及原因；
4. 最近一次归因摘要：各类别计数、E1 / decision_mismatch / E2 top-5 —— 从最新记录的
   `attribution_ref`（`<path>@sha256:<hex>`）校验哈希后读取 `summary.json`，哈希不符 exit 2；
5. 诊断要点（`--diagnosis` 时）：dead / rare 决策、unreachable 阈值位点；
6. 可引用 ID 列表：每条证据旁与汇总段都给出可直接填进 `--insight-refs` 的引用串
   （`rec-<id>` / `bench:<run_id>` / `attr:<path>@sha256:<hex>` / `diag:<path>@sha256:<hex>#<anchor>`）；
7. 纪律提醒。

## 洞察引用（--insight-refs，G2）

`run`、`ledger append` 与 `live-verdict` 接受逗号分隔的 provenance 引用，合并进记录的
`candidate.provenance.insight_refs`（去重）；不可验证的引用直接拒绝（exit 2）：

- `rec-<id>`：台账里已存在的 record_id
- `bench:<run_id>`：基准 run id（仅格式校验）
- `attr:<path>@sha256:<hex>`：归因产物，哈希必须与文件字节一致
- `diag:<path>@sha256:<hex>#<anchor>`：诊断产物，锚定到文件内某个位置

## 诊断与分布校准（D）

两条无需标注的命令，把无标签流量变成提议依据：

- **`diagnose`**：对 `collect-replays --unlabeled` 的原始行做三类体检——
  死决策（从未命中）与稀有决策（有命中但流量占比低于 `--rare-pct`；
  `--expect-rare` 白名单合法稀有决策如安全兜底，只报告不告警）、
  各信号的分位数 / 直方图 / 量程利用率（`--utilization-warn` 标记量程利用不足，
  `--min-sample` 警告样本过少）、以及无流量满足的规则谓词位点（unreachable）。
  产出 `diagnosis.json`（recipe-lab-diagnosis-v1，输入哈希、字节级稳定，
  即 `next --diagnosis` 的输入）与 `diagnosis.md`。
- **`propose-calibrated`**：从 `diagnosis.json` 读一个信号的观测分布，
  在指定分位点（`--quantiles`）或目标流量占比（`--target-share`，等价于分位点
  `1 - share`）提议规则阈值候选；复用规则级提议器，共享 (signal, op, value) 的
  位点一起移动。旁车引用来源 diagnosis；其 config 哈希与 `--base-config` 不符
  直接拒绝（exit 2）。`--from` 限定只移动当前处于该值的位点
  （BUG-3 的 2.5 → 1.75 校准）。

## 守卫与约束（live-verdict）

- **守卫以契约为准（A3）**：地板只来自 `spec.gates`（`{name, metric, floor}`）。
  `--guard-measurements` 文件只带测量值（`name` / `candidate` / `baseline`，由 model_eval
  守卫工具离线产出）；文件里出现 `floor` 视为绕过契约，exit 2。`floor: baseline`
  解析为测量出的 baseline 值。缺测量 / 缺 baseline 值 / 缺 candidate 值 → 该 gate
  未评估，裁决 INCONCLUSIVE；契约未声明的测量名列出并说明，但不参与否决；
  `gates: []` 合法，输出显式 `"gates": {}` 并带 "no guard declared" 提示。
  replay 记录不评守卫，声明的 gate 全部记为 `not_evaluated`。
- **经济约束（E）**：`objective.constraints`（`{name, metric, op, bound, tolerance_pct}`，
  op 为 `gte`/`lte`，bound 为数字或 `baseline`，tolerance 向宽松方向放宽 bound）。
  语义与守卫区分：守卫是安全否决，约束是经济条件（如成本不得回退）。
  已核实的 sr-bench compare 成本字段在 `baseline_targets`/`candidate_targets` 上：
  `total_spend_usd` / `cache_neutral_cost_usd` / `cost_usd` / `evaluation_cost_usd`；
  用量未知序列化为 null → 约束不可评估 → INCONCLUSIVE（绝不记 0，也绝不记违规）。
  约束证据进入裁决输出与台账记录（`constraints` 字段），供报告重放。

## 记录格式（归因输入）

每行一个 JSON 对象（router_replay 适配器产出同此形状）：

```json
{"case_id": "c1", "truth": "math", "matched_decision": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null`（null = 无证据，按证据优先原则不计为错误）
- `family`: 产生该判定的信号家族（`vela_classify` / `vela_decisions` / `d3`），归因按家族分列
- `matched_decision` is the decision the router actually matched (F1 renamed
  the earlier `pred`; reading old records with a top-level `pred` field still
  works).
- `truth` is either the hand-labeled expected decision for that prompt (a
  string), or an object `{"signals": {...}}` carrying per-signal ground-truth
  facts (F2): each fact is a label string (`"domain": "code"`), a number
  (`"difficulty": 3`), or a numeric predicate (`"difficulty": {gte: 3}`).
  Facts drive both the oracle policy's rule evaluation and E1 signal
  judgment.
- Signal evidence (`signal_values` / `signal_confidences` / `decision_tier`)
  is carried verbatim from the replay record. Keys are `<type>:<name>` (e.g.
  `decision:difficulty`); choice-type decision signals store one probability
  per choice under `decision:<name>:<choice>` (e.g. `decision:domain:code`),
  while score-type signals keep the numeric score under `decision:<name>`.
  When reading a signal, values fall back per key from `signal_values` to
  `signal_confidences` (matching the router's own lookup).

## Oracle policy

The oracle policy is declared in the experiment spec under
`attribution.oracle_policy`; attribution never recomputes it from router
logic. Two forms are supported (F2):

```yaml
attribution:
  oracle_policy:
    # form 1 (legacy): decision label -> intended model
    math: local/model-math
    code: local/model-code

    # form 2 (F2): signal-fact rules, evaluated top-down; first match wins
    rules:
      - when: {domain: code, difficulty: {gte: 3}}
        expect: {decision: domain_code_complex, models: [m/pro, m/max]}
      - when: {domain: code}
        expect: {decision: domain_code}
    default: {decision: simple_general}
```

String truths resolve through the label mapping; `{"signals": ...}` truths
evaluate the rules against the truth facts (unmatched and default-less cases
become `policy_gap`). A truth whose signals contradict the record's observed
signal evidence is `E1_signal` (the failing signal and its observed value are
recorded on the row).

## Signal family

`family` on attribution rows is derived from the case's own signal evidence
key type (F2): `classifier:` -> `vela_classify`, `decision:` ->
`vela_decisions`, `projection:` -> `d3`; other key type segments surface
verbatim. Records without signal evidence keep whatever `family` tag the
input record carries (e.g. from `collect-replays --family`).

## 契约要点

- `objective.replay_metric` 是 replay 层唯一排序指标（A1）；旧 `evaluation` 段已退役，
  加载即报错（A5）。
- `objective.constraints` 声明 live 层经济约束（E）；守卫地板只来自 `gates`（A3）。
- `attribution.oracle_policy`（F2）取两种形态之一：label → model 映射，或
  `{"rules": [{"when": {<信号>: <条件>}, "expect": {decision, models?}}, ...],
  "default": {...}?}`。条件按标签串、有限数值或 `gte|lte|gt|lt` 谓词映射匹配信号事实
  （路由器语义：`gte` 在值小于 bound 时不命中）；首条命中的规则生效，缺省兜底
  `default`，两者都不覆盖的真值是 policy gap。
- `workload` 声明三分 split（B）时必须给出 `split_rule` 文本且三分互不相交；
  旧的单数据集形态按 dev-only 处理并告警。
- `budget.max_minutes_per_trial` 覆盖就绪等待 + 请求循环 + replay 拉取全程（A5）。

| class | meaning |
|---|---|
| `correct` | routed as intended and answered correctly |
| `E1_signal` | a truth signal contradicts the record's observed signal evidence (needs signal-level truth: truth = `{"signals": ...}`; F2) |
| `decision_mismatch` | matched decision != expected decision, judged without usable signal-level truth (F1; with fully verified signal truth a decision miss is `E2_policy`) |
| `E2_policy` | decision or routed model deviates from the oracle-policy expectation, answer wrong |
| `E3_capability` | routing exactly as intended, answer still wrong |
| `suboptimal_routing` | policy deviation but still a correct answer (counted, never an error) |
| `no_evidence` | answer correctness unknown; never an error |
| `policy_gap` | truth (label or signal facts) not covered by the declared oracle policy |

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
