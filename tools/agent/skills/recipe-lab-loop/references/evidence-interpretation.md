# Reading recipe-lab evidence

## Verdicts

- `RANKED` (replay tier): ordering of unverified numbers. Never quote a
  RANKED score as a capability claim.
- `PASS` / `FAIL` (live tier): evidence complete, guard floors evaluated.
  `FAIL` with "below min" means the candidate did not beat the baseline.
- `INCONCLUSIVE`: incomplete evidence (timed-out trials, missing metrics,
  unfinished guard evaluations, ambiguous compare payloads). Fix the
  evidence problem; do not treat it as a regression.

## Misroute classes (attribution report)

- `E1_signal`: the signal model disagreed with ground truth. Fixes differ by
  signal family: fixed-head classifiers take threshold calibration or
  retraining; dynamic decision-question surfaces can also take question-text
  edits (a later proposer).
- `E2_policy`: signals right, routing wrong - decision table, weights, or
  fallback. Script-searchable.
- `E3_capability`: routing was right, the model still failed. Not a routing
  problem; may motivate a new signal (router-contract-change).
- `SUBOPTIMAL`: correct model but a better one was reachable. Track, do not
  punish.
- `NO_EVIDENCE`: missing answer-bank coverage. Collect evidence; never score
  it as wrong.

## Honesty rules

- Metrics are descriptive medians/ranges; statistical significance is not
  established anywhere in recipe-lab output.
- Cross-dataset comparisons are forbidden: different `dataset_hash`, no
  comparison.
- A candidate that only wins where evidence is missing has won nothing.
