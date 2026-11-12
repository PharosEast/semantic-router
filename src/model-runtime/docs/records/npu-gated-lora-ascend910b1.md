# Gated LoRA on Ascend 910B1: per-adapter passes re-read the whole batch

Issue: the vela3_d3 adapters (vela3-lite/-mini/-flash) run one adapter per
pass. Precedents: `npu-fla-chunk-ascend910b1.md` (the same host and model),
`decision3-batching.md` (the batching the gated rows never pack).

- **Date:** 2026-11-12.
- **Device:** Ascend 910B1 (Huawei Cloud EulerOS 2.0 aarch64, CANN 8.5.1,
  torch 2.10.0, torch_npu 2.10.0.post4), one card of a shared host.
- **Model:** `vllm-sr/vela3-mini` at pin `bbc0206e` (d3-mini base
  `566da0ea`), 32 layers, hidden 2560, 200 LoRA-wrapped Linears for the
  `decision`/`decision_df`/`decision_optin` adapters.
- **Code:** main `9968ac62a`.

## What was measured

Three-question requests through `golden_surface("decisions", ...)`, with hooks
counting real `engine_model.forward` calls and `GatedLinear.forward` calls:

| scenario | rows | backbone forwards | median |
| --- | --- | --- | --- |
| 1 question, 1 adapter | 1 | 1 | 557 ms |
| 2 questions, 2 adapters | 2 | 2 | 4,186 ms |
| 3 questions, 3 adapters | 3 | 3 | 9,308 ms |

Controlled for the row count (the same three questions, only the pass list
changed):

| passes | median | GatedLinear calls |
| --- | --- | --- |
| 0 (base rows only) | 3,054 ms | 200 |
| 1 | 3,104 ms | 200 |
| 3 | **9,308 ms** | 600 |

## Findings

1. **A pass re-reads the whole batch.** `GatedDecisions._readout` looped
   `for name in order: super()._readout(items, ids, mask, last, ...)` with the
   full batch and selected the pass's rows afterwards, so each adapter re-read
   every other adapter's rows to discard their results. Three adapters measured
   exactly 3.00x the one-pass work.
2. **The adapter kernels themselves are cheap.** One pass over the base rows
   costs 3,054 ms and with the `decision` adapter 3,104 ms: **+50 ms, 1.6%** of
   the pass. A microbenchmark at the served shapes (rank 16, one row) measured
   the plain Linear at 0.086 ms and the gated forward at 0.29 ms (masked
   0.38 ms), and the cost is **flat in rows (1/2/4/8 all alike)** — kernel
   launches, not FLOPs. Micro-optimising this path (BF16 factors, folding the
   scaling into `B`, fusing `where`) is worth at most that 1.6%.
3. **Padded rows cost 1.83x.** Gated rows never pack (`_packs` returns False),
   so three rows in one forward cost 3,054 ms against 1,671 ms for three
   single-row requests.

## Change

`GatedDecisions._readout` runs each pass over the rows it answers: the base
rows ride the first adapter's pass (its gate leaves them on the base's values,
their answer either way), and every later pass runs its own rows alone.
`tests/test_vela3_d3_decisions.py` follows: a gated row's answer is asserted
against the one-row forward, and the first pass's gate covers the base row
plus its own adapter's row.

The golden path is unchanged: a reference forward is one request alone, one
row and one adapter, where the old `(True,)` and the new `None` gate both mean
"every row runs this adapter".

## Result

The same three-question request, back-to-back on one card:

| | before | after |
| --- | --- | --- |
| median | 9,308 ms | **3,216 ms (2.89x)** |
| backbone forwards | 3 | 3 |
| answers | — | **0 flips**, max \|dp\| 1.55e-5 |

The 1.55e-5 divergence is the BF16 rounding of the batch-shape change
(a row computed alone instead of inside a padded three-row batch), 0.08% of
the npu readiness gate's `GPU_TOLERANCE` (0.02) and three orders under
`decision3-batching.md`'s measured 0.059 for batched-against-alone.

The residual against the 5.57x the row accounting predicts (3 x 557 ms) is
padding: the three questions differ in length, so the longest sets each
pass's width. Bucketing rows by length is the next lever; folding each
adapter into the base weights (`W' = W + scaling * B @ A`) removes the
remaining 1.6% but costs one weight copy per live adapter and an accuracy
record.

## Reproduction

Scripts and data on the 910B1 host under
`sr-npu/gated-lora/{scripts,data,logs}`: `gated_microbench.py` (per-module
breakdown), `gated_end_to_end.py` (pass counting and the controlled
pass-count experiment), `gated_capture.py` (the answer capture the diff
uses), `data/{microbench,module_shapes,end_to_end,answers_baseline,
answers_patched}.json`.
