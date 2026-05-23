# artifacts/

Committed reproducibility artifacts so the README references work without re-running the benchmark.

## Contents

| File | What |
|---|---|
| `headline.png` | The headline 4-policy bar chart with Wilson 95% CIs |
| `seed_manifest.json` | Full benchmark configuration: seeds, magnitudes, model versions, policy definitions |
| `l0_report.json` | Output of `verify/l0_check.py` — confirms SAE encoder produces L0 ≈ 99 (expected 91) |
| `baseline_results.jsonl` | 24 runs of the baseline policy |
| `targeted_results.jsonl` | 24 runs of the targeted policy |
| `wrong_sign_results.jsonl` | 24 runs of the wrong-sign control |
| `sample_trajectory_baseline.jsonl` | One full trajectory of baseline failing on promo_held_001 |
| `sample_trajectory_targeted.jsonl` | One full trajectory of targeted succeeding on promo_held_001 |

## To regenerate from scratch

```bash
make verify            # SAE smoke test
python -m verify.l0_check                       # L0 sanity
python -m verify.feature_drill                  # contrast discovery
python -m verify.tune_deltas                    # magnitude calibration
python -m verify.step0_calibration              # step-0 win discovery
for p in baseline random wrong-sign targeted; do
  python -m bench.runner --policy $p --tasks shopgym/tasks/held_out.json --trials 3 --limit 8
done
python -m bench.analysis --plot                 # writes headline.png
```

Total wall time: ~90 min on a Modal L40S after first cold start.
