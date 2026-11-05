# recipe-lab

Closed-loop experiments for semantic-router routing recipes（RSI 工作台）。

本层覆盖 replay-tier 闭环：冻结题集身份（A4）→ 题集三分（B）→ 答案库与归因 →
候选提议（信号级 / 规则级 / 分布校准）→ replay 层评估入台账（A1/A5）→ 台账检索与洞察引用（G1/G2），
以及 live 层终裁 `live-verdict`（守卫以契约为准 A3、洞察引用 G2、经济约束 E）
与端到端编排 `verify`（I）。另含无需标注的分布诊断（D）、oracle policy 的
规则/默认形态（F2）、replay 热加载（H）与成本代理（E2）。

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
   全部被拒则 exit 2。`--reload hot` 用单路由 + `PUT /api/v1/config` 切换候选，
   失败回退 restart 并记录（H）。spec 带 `cost_model` 时额外产出
   `routed_cost_index` 成本代理：声明约束被代理评估，违约候选带
   `constraint_warning` 且不参与 `best_replay`（E2）。
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
| `observe-labels` | C2 冷启动观测（遗留 domain 路径）：种子配置 + request-id 打标探测 → labels-<split>.jsonl（case_id/router_label/confidence，未命中记 other）；`--classification-config` 指定分类源配置，其 modules/embeddings 注入种子（Y2 同源） |
| `observe-lanes` | C2 冷启动观测（模板路径，sub-issue-8 §2.2）：`vllm-sr recipe builtin list` 读模板 lane → `recipe builtin init` 把模板绑到占位模型生成观测配置（决策模型由模板钉住，无需 Y2 一致率检查）→ request-id 打标探测 → lanes-<split>.jsonl（case_id/lane/confidence）+ summary（lane_coverage、模板哈希、init 命令） |
| `profile` | C3 模型画像：`--labels`（遗留 domain 路径）或 `--lanes`（模板路径）二选一分组 + 调用 model_eval 脚本直连各模型 → profile.json（含区间/成本/质量门；lane 模式不合并稀有 lane，低于 `bootstrap.min_lane_cases` 的记为 unmeasured_lanes，未观测到 lane 的记 unobserved） |
| `bindings` | C4+C5 冷启动候选（模板路径，sub-issue-8 §2.2）：路由空间检查（无空间则出"不建议路由"报告、不生成候选）→ 每个 λ 一份 bindings（能力过滤 → 正确率 − λ×归一化成本排序 → 满足 `minimum_candidates`；未测量 lane 按默认规则并标注"未经实测"；能力无法满足的 lane 必须 `--exclude-decision` 显式排除）→ `recipe builtin init` 以 spec 的 providers_config 生成候选（评估副本关闭响应缓存）→ `config validate` 通过才落盘；旁车记模板哈希/逐 lane 依据/未实测与被排除 lane |
| `bootstrap` | C4+C5 冷启动候选（**遗留 domain 模板路径**，sub-issue-8 §2.5，需显式 `--legacy-domain-template`，不计入完成条件）：每个 λ 调真实 `result_to_config.py` 出骨架（只吃 calib，Warning 即拒绝），补全成可运行配置（契约端点/按第一名分组的 `route-<模型>` 决策/流量优先级/缓存与 tools 关闭）并逐个过 `config validate` 后才落盘；Y2：候选的分类节（modules/embeddings）来自 `--classification-config`（与 observe-labels 的种子同源，骨架副本被丢弃），随后 dev split 经种子与每个候选各观测一次，标签一致率 < 99% 拒绝（exit 2），一致率与分类源哈希写入 provenance 旁车 |
| `bank` | 构建 / 导入 / 应用 per-case/per-model 答案库（`contains` / `mcq` 判卷，I；`import` 直接转换 C3 画像的 dev judged.jsonl，C6） |
| `diagnose` | 无需标注的分布诊断：死/稀有决策、信号分布、不可达位点（D）；agentic 行携带相位/agent 标记时追加 "Agentic session evidence" 段（分相位分布、子 agent 分开保持、失败披露，AG4）；`--agent-requests` 以 probe 的子 agent 清单按请求 id join 保护轨迹身份哈希，判定子 agent 是否独立保持（Z9，不可 join 或空清单保持 unverifiable） |
| `profile-agentic` | AG5：汇总各模型单模型 sr-bench run 报告 → agentic 画像（成功率+区间/每任务成本实际与对照/调用数/时延/截断率，缺失字段显式 missing） |
| `bootstrap-agentic` | AG5/Z10 agentic 冷启动：以 agent recipe 模板 + 画像 + 客户端身份契约生成候选。选型走共用核心（`bindings.plan_lane` 的未测量 lane 分支：能力过滤 + λ 调整总体分排序 + "默认规则，未经实测"标注——agentic 画像只测模型不测 lane，`--lambda` 默认 0；隐私/安全 → local_models 且 bypass，不参与排序）；claude-code 写 session 作用域 + x-claude-code-session-id/agent 头，其余写会话双头；自适应关、进度门 observe、candidate_requirements 声明能力；旁车记录每条 lane 的依据与画像证据 |
| `run-agentic` | AG2/Z14 agentic 驱动：headless Claude Code 逐题跑任务文件（SWE-bench 行），每题独立工作目录与会话；本地转发代理记录原生身份头（W2：x-claude-code-session-id，子 agent 请求带 x-claude-code-agent-id）并为无 request-id 的请求注入确定性 `x-request-id`（Z9 join 键）；端点/模型/claude 路径/克隆模板全部参数化；产物 `results/`、`requests/`、`agent-requests.json`（diagnose --agent-requests 直接可用的子 agent 清单）、`task-results.json`（sr-bench 同形逐任务行，judged 前 correct 为 null） |
| `judge-agentic` | AG2/Z14 判卷：对 run-agentic 目录调 SWE-bench 官方 harness（命令模板全参数化，占位符 {predictions_path}/{dataset_name}/{run_id}/{max_workers}/{log_dir}/{report_path}），输出 `judge-results.jsonl`（sr-bench 同形：correct/score/cost_usd/call_count/latency_s）与 `judge-summary.json`；报告不覆盖的任务保持 correct=null，绝不记 0；判卷可与生成分开批量执行 |
| `propose-thresholds` | 扰动单个决策信号阈值 |
| `propose-rule-thresholds` | 移动一个信号的所有规则级谓词位点（C） |
| `propose-calibrated` | 按观测分布分位点提议规则阈值（D） |
| `propose-session-policy` | AG6/Z6：按原因分类的单变量会话保护候选（无保护切换→补身份头映射，W1/W2 类；锁外普通切换→提 stability_weight 与 switch_margin；子 agent 未分开保持→加 agent 身份头）。决策漂移超限→不调旋钮，交回阈值校准（propose-calibrated）；被拒请求→交回"候选集与锁的关系"检查；身份头已配仍无保护→交回客户端身份头修复（均以 `handoffs` 回报，不产配置）。候选先过 AG3 route-replay 门槛再进 live |
| `run` | replay 层评估候选 → 台账 RANKED（A1/A5/C/G2；`--reload restart\|hot` H；`cost_model` 成本代理 E2） |
| `ledger` | 台账 append / show / top（G1/G2） |
| `verify` | 驱动 sr-bench 端到端：preview + AB/BA 多轮 → 三态裁决（I）；`driver: model_eval` 工作载改走 model_eval 驱动（C7） |
| `live-verdict` | 比较两个现成 sr-bench run 出三态裁决（A3/G2/E，frozen_test 仅 `--final`） |
| `report` | 只从台账重放的运行报告（G1） |
| `provenance` | recipe-lab-run-v1 manifest（记录 ID、台账哈希、bench_refs） |
| `record-draft` | 从最近一条 PASS live 记录生成晋升草稿 |
| `bundle` | 成品打包：live PASS + `final: true` 的记录 → 可评审目录（C9，见下） |
| `next` | 导出下一轮 agent 上下文（G3；不可信材料头 + 可引用 ID） |

常用示例：

```bash
# 校验并预览实验契约（不执行任何实验）
recipe-lab plan --spec examples/spec.example.yaml

# 三分题集（同题集 + 同参数 => 字节级一致输出）
recipe-lab split --cases labeled.jsonl --output-dir splits/

# C2 冷启动观测：真实 router 自己的域标签（summary 的 label_coverage 写进 provenance）
# --classification-config：种子与候选共享的分类源（冷启动传维护版 config/config.yaml，Y2）
recipe-lab observe-labels --spec spec.yaml --split calib --out observe/ \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py \
  --classification-config config/config.yaml

# C2 模板路径（sub-issue-8 §2.2）：recipe builtin init 生成观测配置，
# 模板钉住的决策模型决定 lane；spec 需声明 template（bundle/recipe）
recipe-lab observe-lanes --spec spec.yaml --split calib --out observe-lanes/ \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py

# C3 模型画像：调 model_eval 脚本直连各模型，profile.json 含脚本哈希与完整命令行
# --labels（遗留 domain 路径）或 --lanes（模板路径）二选一
recipe-lab profile --spec spec.yaml --labels observe/ --splits calib,dev \
  --out profile/ --model-eval-dir src/training/model_eval

# C4+C5 模板路径：路由空间检查 → 每个 λ 一份 bindings → recipe builtin init
# 生成候选（过 config validate 才落盘）；无路由空间时输出报告、不生成候选
recipe-lab bindings --spec spec.yaml --profile profile/ \
  --lambdas 0,0.1,0.2,0.3 --out cands/ \
  --vllm-sr-bin vllm-sr --model-name vllm-sr/auto

# C4+C5 遗留 domain 模板路径（需显式 --legacy-domain-template；sub-issue-8 §2.5）
# 产物：scaffold-lambda-<λ>.yaml、candidate-lambda-<λ>.yaml、
#       candidate-lambda-<λ>.provenance.json（proposer/λ/逐标签表/哈希/命令）
# Y2：--classification-config 与 observe-labels 同源；bootstrap 随后对 dev split
#     跑标签一致率检查（< 99% exit 2），需要 --router-bin/--mock-backend 启动检查栈
recipe-lab bootstrap --legacy-domain-template --spec spec.yaml --profile profile/ \
  --model-eval-dir src/training/model_eval --lambdas 0,0.1,0.2,0.3 \
  --out cands/ --validate-bin vllm-sr \
  --classification-config config/config.yaml \
  --router-bin build/bin/semantic-router --mock-backend scripts/mock_backend.py

# C6 答案库：直接转换画像的 dev judged.jsonl（不再打端点），
# 每条带 case_id/model/answer/answer_ok/tokens
recipe-lab bank import --spec spec.yaml --profile-dir profile/ \
  --split dev --output bank.jsonl

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

# 端到端终裁（I）：驱动 sr-bench —— preview + 3 轮 AB/BA + compare + 裁决
recipe-lab verify --spec spec.yaml --baseline-run-id run-a \
  --candidate cand-001.json --repetitions 3 --ledger ledger.jsonl
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
  replay 层的代理评估见下节（E2）。
- **候选配置绑定（Y4）**：两个 run 的 `manifest.targets` 里的 live MoM target
  都必须带冻结 `config_hash`，且相互一致；裁决结果与台账
  `candidate.config_sha256` 记录这份哈希（`sha256:` 前缀）。`--final` 晋升
  裁决在台账写 `final: true`（live 层专属；bundle 以此识别晋升记录）。

## verify（驱动 sr-bench 的端到端终裁，I）

`verify` 不比较两个现成 run，而是自己驱动 sr-bench：

1. 从终态 `completed` 的 baseline run 提取冻结 manifest（`benchmark show`）；
2. 跑一次 preview candidate-plan——纯诊断：路由分布写进台账 notes，
   preview 失败只记为诊断原因，绝不否决裁决；
3. 跑 `--repetitions` 轮 AB/BA——每轮重跑冻结 baseline manifest + 新建
   candidate-plan，然后逐对 compare；
4. 各对指标取中位数/范围出裁决；任何一对不完整 → INCONCLUSIVE。

输出与 `live-verdict` 同形的 JSON，退出码也一致：PASS 0、FAIL 1、
INCONCLUSIVE 与工具/用法错误 2（`verdict_exit_code`）。`--candidate-targets` 指定服务候选的 sr-bench MoM target，
缺省回退到候选 JSON 的 `srbench_targets`。守卫与约束语义与 `live-verdict`
一致（A3/E）；`--split frozen_test` 同样只留给 `--final` 晋升裁决。
候选配置绑定（Y4）：verify 先在 sr-bench 注册表核对 `--candidate-targets` 的
每个 MoM target 都已注册且带冻结 `config_hash`，多 target 哈希不一致即拒绝；
结果与台账 `candidate.config_sha256` 记录该哈希，`--final` 产出带
`final: true` 的台账晋升记录。

### agentic 工作载的 replay 证据（Z4/Z5）

`kind: agentic` 工作载的稳定性硬门槛以臂自身路由器的 replay 记录为唯一
证据源：被拒请求、工具循环中断（事件计数），以及工具循环切换的原因拆分
（Z5）。每个完成的 MoM 臂 run 结束后，verify 按该 run 的会话身份
（`<run_id>-<digest>`，sr-bench engine 逐任务派发）从该臂路由器的管理
API 拉取 replay 并过滤，门槛消费与 continuity 证据同源的"最后完成对"。
因此 `--candidate-management-endpoint`（候选臂路由器）必填；
`max_tool_loop_interrupted` 默认钉 baseline，还需
`--baseline-management-endpoint`（baseline 臂路由器）——缺失即 CLI 快速
失败，绝不花算力跑注定 INCONCLUSIVE 的裁决。拉取失败只记诊断：证据缺失
保持未评估，绝不是 PASS。单模型参照臂不经过路由器、没有 replay：没有
可比 baseline router 的比较（冷启动）应把 `max_tool_loop_interrupted`
声明为数值上限（如 0），不要钉 baseline。结果的 `agentic` 块以
`replay_evidence` 记录两侧各判了哪个 run 的行（run id、端点、行数），以
`switch_evidence` 回显候选臂切换的原因拆分。

Z5（上游 #5567）：分类漂移到另一决策时工具循环锁释放，漂移迁移是设计内
行为，原始切换数不能一刀切判 0。切换门槛按原因拆成两个：
`max_unprotected_tool_loop_switches`（默认 0）= continuity 的工具循环
切换总数 − replay 保护轨迹标记为 `decision_drift` 的释放数（身份缺失 /
bypass / observe 下的切换只在 continuity 可见，落在残差里）；
`max_decision_drift_switches_per_task`（默认 0.2）= 漂移释放数 / replay
会话数（一个会话即一个任务），超限提示去做阈值校准
（propose-calibrated），不是路由回归。

### model_eval 驱动（C7）

工作载声明 `driver: model_eval` 时，verify 完全绕开 sr-bench：每个臂都是
一次 `multi_benchmark_eval.py` + `mcq_judge.py`——baseline/candidate 两臂
分别打各自配置的 router 端点，S/W 参照臂直连 `spec.models` 里对应模型的
端点。`--tasks` 必须是契约声明的 `model_eval_tasks` 渲染（`cases
to-model-eval` 生成的 split 级文件；契约里同时声明 `model_eval_tasks`
与 `model_eval_tasks_sha256`，verify_workload 既核哈希又核 task_id 与
cases manifest 一一对应），题集改写不再接受外部成品：`--labels` 传入
C2 的 labels 文件，稀有标签合并与未覆盖题归 other 都由驱动内部按
`min_label_cases` 完成（改写后题集、labels 哈希与合并明细都记录在
结果里）。逐题对错交给与 sr-bench 驱动相同的 M2/M3/M4 机器：M2 配对
统计走 `compute_paired_stats`（sr-bench compare 字段形状），M3 证据质量
用同一组比率键与上限，M4 参照臂按（改写后题集哈希, 别名）缓存。准确率
按计划分母计（缺答/漏答计错）；某个臂崩溃是操作性错误（脚本可续跑，
exit 2），臂产不出 judged 行是证据不完整（INCONCLUSIVE）。必填：
`--tasks`、`--labels`、`--candidate-config`、
`--candidate-management-endpoint`、`--baseline-router-endpoint`、
`--candidate-router-endpoint`、`--model-eval-dir`；sr-bench 相关旗标在此
模式下被忽略，`--baseline-run-id`/`--baseline-manifest` 会被拒绝。

**候选配置绑定（Y4）**：`--candidate-config` 的 sha256 写入结果与台账
`candidate.config_sha256`，且候选 router 的管理 API（`GET
/api/v1/config/hash`）必须确认一份已激活（`activation_status: active`）、
source hash 与该文件一致的配置——路由器还在跑旧文档时任何臂都不派发，
直接 exit 2，绝不给错配置打分。管理端点因此必填（随机混合 R 的 replay
查询同用它）。

**成本匹配随机混合 R（C8）**：驱动由 candidate 臂的 replay 记录得到
路由分布 p_m——router 场景的 record 按 prompt 与实际路由决策 join
（见「记录格式」），按 `spec.models` 的 served name 映射到别名——再按
份额混合参照臂的逐题对错与单例成本，得到随机混合 R 的期望准确率/成本，
并给出 C−R 的配对 bootstrap 区间（每次重采样 R 都按份额重新分配）。R 以
`ref:random_mix` 角色进入裁决输入（约束 `bound` 可用）、追加一条
`ref:random_mix` Pareto 行，区间写入结果的 `random_mix` 节（P3 的
「准确率差区间下界 > 0」由此计算）。缺证据时（candidate records 无
replay 路由结果、路由到 `spec.models` 之外、参照臂缺失）R 不可评估并
给出原因，绝不猜值。

### Bundle（C9，成品打包）

`recipe-lab bundle --spec S --ledger L --record rec-... --out bundle/ --config
FINAL.yaml --skeleton SCAFFOLD.yaml --candidate-sidecar PROVENANCE.json
--profile PROFILE.json` 把一条记录的验收证据打包成可评审目录。门槛是硬性的：
记录必须是 live PASS 且携带 `final: true` 晋升标记（`verify --final` 写入，
bundle 直读该字段，不做契约反推）；打包的配置就是被终裁的那份——`--config`
的 sha256 必须与台账记录 `candidate.config_sha256` 一致，`--skeleton`/
`--profile` 的 sha256 必须与候选旁车（bootstrap 写的 `<候选>.provenance.json`，
`--candidate-sidecar` 指定）引用的 scaffold/profile 一致；旁车还必须携带 Y2 的
`classification.projection_sha256`，与 `--config` 自身的分类投影哈希
（deployments/system/modules.classifier/modules.prompt_guard/embeddings）一致，
即打包配置与被评候选的分类行为逐字节同源；且 `--config` 能通过
`<validate-bin> config validate`（`--validate-bin` 可换，约定同 bootstrap）；
任一不满足即 exit 2，什么都不写。产物对同样输入字节级稳定：

| 文件 | 内容 |
|---|---|
| `config.yaml` / `profile.json` | 输入文件的字节级拷贝 |
| `validate.txt` | `vllm-sr config validate` 的输出（校验失败则整体拒绝） |
| `config.diff` | 相对 `--skeleton` 骨架的 unified diff |
| `report.md` | 裁决细节（配对区间、约束、路由分布）、活动台账表、按 workload 计数、证据质量、全部假设、局限，以及关闭缓存和 tools 的说明 |
| `ledger.jsonl` | 本次活动（同 dataset_hash）的台账摘录 |
| `provenance.json` | 全部哈希：题集、骨架、配置、画像、台账、run id，外加 `--router-commit`、`--model-revision NAME=REV`、`--model-eval-script`（缺省记为 null，绝不猜值），以及 Y2 分类绑定（`classification.projection_sha256` + 分类源 `source_sha256`，只记哈希不记路径） |
| `PROMOTION.md` | record-draft 输出（sub-4 的「草稿一例」） |
| `REVIEW.md` | 空模板，V5 评审时填写 |

写盘前扫描全部产物的私有内容（个人用户目录、非本机 IP 字面量 URL，可用
`--private-pattern REGEX` 追加），命中即失败且不回显命中内容。验收证据的
归档目录放在仓库之外，不进公开仓库。

### 冷启动（C10，从零生成配置）

没有可改的存量配置时，从 n 个后端模型加一份带标签的选择题题集出发，
`examples/cold-start/cold-start.sh` 按顺序串起全流程：

```bash
recipe-lab split ...             # 切 calib/dev/frozen_test；hash 先锁定题集指纹
recipe-lab observe-labels ...    # 真实 router 在 calib 上实际打出的 domain 标签
recipe-lab profile ...           # model_eval 直连各模型：骨架输入、答案库、token 成本
recipe-lab bootstrap ...         # 每 λ 一份骨架 + 补全成可用候选（逐个过 config validate）
recipe-lab bank import ...       # 画像的 dev 判卷结果转答案库（不再打端点）
recipe-lab run ...               # replay 排序，RANKED 进台账
recipe-lab verify --split frozen_test --final ...   # live 终裁
recipe-lab bundle ...            # 成品打包
```

脚本参数全部来自环境变量（`CASES`/`SPEC`/`VERIFY_CANDIDATE`/
`BUNDLE_CONFIG`/`BUNDLE_SKELETON`，旁车默认取 `<BUNDLE_CONFIG%.*>.provenance.json`
可用 `BUNDLE_SIDECAR` 覆盖，外加各步 `*_FLAGS` 透传，见脚本头注释），
不含任何私有路径；阶段按输入齐备情况依次执行，缺什么就提示下一步设置什么。
失败行为：profile 校验模型别名可往返、bootstrap 的 Warning 即拒绝、候选先过
`config validate` 才落盘，任一步失败即退出且不产出后续产物；verify 需要真实
端点（`VERIFY_FLAGS` 携带），bundle 拒绝非 live PASS、无 `final` 标记或哈希
不绑定的记录（见上节）。
frozen_test 只被 `verify --final` 触碰一次，调参与排序永远只看 calib/dev。

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
- 顶层可选 `cost_model`（E2）：served 模型名 → 相对成本（必须有限正数）。
  声明后 replay 层产出 `routed_cost_index`（已作答样本按路由模型成本均值；
  未知模型成本绝不记 0）并代理评估成本约束；不声明则成本只在 live 层裁决。
  C6：`cost_basis.source: provider_pricing` 且声明了 `assumed_pricing` 费率时，
  replay 成本模型自动从费率推导（prompt+completion 单价和相对最便宜的正值模型
  归一；免费模型为 0），无需再写 `cost_model`；两者同时声明是契约错误。
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
