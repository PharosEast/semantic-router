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
{"case_id": "c1", "truth": "math", "pred": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null`（null = 无证据，按证据优先原则不计为错误）
- `family`: 产生该判定的信号家族（`vela_classify` / `vela_decisions` / `d3`），归因按家族分列

## 测试

```bash
pytest
```
