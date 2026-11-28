# Frozen defense analysis

Run from `quickstart-pytorch/` with the project environment and its existing
Matplotlib dependency installed:

```bash
python -m analysis.analyze_defense
python -m unittest discover -s tests -p 'test_analyze_defense.py'
```

The script reads only these explicit validation-only executions, each with
seeds 42-51 and four conditions per seed (FedAvg and coordinate-wise median,
each clean and attacked):

| Dataset | Input | Local epochs |
|---|---|---:|
| OrganAMNIST | [defense generation-v1/protocol-v3/run-2](../results/defense/generation-v1/protocol-v3/run-2/) | 1 |
| OrganCMNIST | [defense generation-v1/protocol-v8/run-1](../results/defense/generation-v1/protocol-v8/run-1/) | 3 |
| OrganSMNIST | [defense generation-v1/protocol-v8/run-2](../results/defense/generation-v1/protocol-v8/run-2/) | 3 |

All use 20 clients, five rounds, batch size 32, learning rate 0.01 and, for
the attacked condition, four clients flipping all kidney-left labels to
kidney-right. The datasets have different protocols and must not be pooled.
The historical 0.1 experiments and earlier pilots are excluded; incomplete
or later expansions are not selected automatically. See
[defense results](../defense_results.md) for the frozen protocols, floors,
limitations and run-level verification.

The script checks the exact seed/condition matrix, record metadata, training
validity, relevant configuration, validation metrics and matching partition
summaries and attack selections. It rejects duplicate or missing records and
non-null test results. Matching partition summaries are not proof of identical
individual indices; that also relies on the seeded implementation and recorded
environment. Source paths and SHA-256 hashes identify the exact input records;
the original manifests remain alongside those records. The analysis does not
retrain, load checkpoints, or independently re-evaluate model predictions.

## Metrics and files

All metrics are from round 5 of the shared validation set. Accuracy is the
fraction correctly classified; targeted error is the fraction of true
kidney-left images predicted as kidney-right; kidney-left recall is the
fraction correctly classified within that class. Macro recall is the
unweighted mean of the 11 class recalls. Loss is mean per-image cross-entropy.
Rates are fractions (multiply by 100 for percentages); differences are
fractions (multiply by 100 for percentage points), except for loss.

For each seed and metric, change = attacked minus clean within an aggregation;
interaction = median change minus FedAvg change. A negative targeted-error
interaction favors median. Summaries report arithmetic means and **sample**
standard deviations across seeds, calculated on the per-seed values and
paired differences, not differences of standard deviations. These are
descriptive seed variability, not confidence intervals or test-set estimates.

Generated files live only in [outputs/](outputs/) and are overwritten on
rerun; original experiment records and checkpoints are never modified:

- `sources.csv`: condition, seed, record path and SHA-256.
- `paired_by_seed.csv` and `metric_summary.csv`: the five metrics, all four
  condition values, paired changes and interactions.
- `class_recall_by_seed.csv` and `class_recall_summary.csv`: the same paired
  quantities for each class, including validation support.
- `targeted_error_conditions.png` / `.pdf`: per-seed clean-to-attacked
  lines and aggregation means.
- `targeted_error_interaction.png` / `.pdf`: per-seed interactions and their
  mean. Figures label each dataset and execution separately.

The validation set informed model and protocol selection. These summaries
describe only the specified attack and do not establish test-set or clinical
performance. In particular, OrganSMNIST is exploratory and has severe
seed-dependent clean femur-right recall despite passing mean clean floors.
