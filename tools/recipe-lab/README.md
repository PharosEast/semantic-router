# recipe-lab

Closed-loop experiments for semantic-router routing recipes（RSI 工作台骨架）。

设计文档维护在仓库外（recipe-lab 设计笔记：umbrella + 6 个 sub-issue；本目录当前覆盖 Sub-issue 1 的归因部分 + 契约/台账最小实现）。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## CLI

```bash
# 校验并预览实验契约（不执行任何实验）
recipe-lab plan --spec examples/spec.example.yaml

# 误路由归因：E1 信号错 / E2 策略错 / E3 能力错（+ 次优路由、无证据）
recipe-lab attribute --spec examples/spec.example.yaml \
  --records examples/records.example.jsonl --output-dir out/

# 台账：追加（契约校验）与查询
recipe-lab ledger append --file ledger.jsonl --record record.json
recipe-lab ledger show --file ledger.jsonl --tier replay --json
recipe-lab ledger top --file ledger.jsonl --metric-key capability.score
```

## 记录格式（归因输入）

每行一个 JSON 对象（router_replay 适配器后续产出同此形状）：

```json
{"case_id": "c1", "truth": "math", "matched_decision": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null`（null = 无证据，按证据优先原则不计为错误）
- `family`: 产生该判定的信号家族（`vela_classify` / `vela_decisions` / `d3`），归因按家族分列

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

集群级运维脚本（数据集批量下载、长时间循环压测、作业收尾）不属于本仓库，放在仓库外私有目录维护。

