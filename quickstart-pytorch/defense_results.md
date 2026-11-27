# OrganAMNIST median defense pilot

## Status and protocol

The validation-only pilot completed 20 runs: seeds 42–46, each with clean and
poisoned FedAvg and clean and poisoned coordinate-wise median. All runs
reported 20 successful clients in each of five rounds, finite results, and
verified disjoint, complete partitions. The four conditions for each seed
used the same partition; the poisoned conditions selected the same client.
Every run retained five global checkpoints. Test evaluation was disabled.

The attack changed 100% of one client's kidney-left (class 4) training labels
to kidney-right (class 5) in every round. All runs used the existing CNN,
20 clients, five rounds, one local epoch, learning rate 0.1, and batch size
32. Median independently sorts each parameter coordinate and averages the
10th and 11th of the 20 values, giving each client one vote. FedAvg weights
client parameters by training-example counts.

Results and the environment/source-hash manifest are in
[`results/defense/generation-v1/protocol-v1/run-1/`](results/defense/generation-v1/protocol-v1/run-1/);
models and round checkpoints are in the matching
[`models/defense/generation-v1/protocol-v1/run-1/`](models/defense/generation-v1/protocol-v1/run-1/).
Both trees group matched runs in `fedavg/` and `median/` subdirectories beneath
the execution. The original pilot outputs were moved into these subdirectories
without retraining; only artifact references and the layout metadata in the run
records changed. Model files, plots, partition manifests, metrics, and the
original execution manifest were preserved.
The manifest records the starting Git revision and working-tree status: this
pilot ran with uncommitted implementation changes, identified by their source
hashes. The Git revision alone is not the exact code used in the pilot, and
hashes identify source contents but cannot reconstruct them. The environment
and deterministic settings support reproducibility but
do not guarantee bitwise-identical results across hardware or software.

## Round-5 validation outcomes

Targeted error is the fraction of true kidney-left validation images predicted
as kidney-right. Each seed has 568 such images. Values below are percentages.

| Seed | FedAvg clean | FedAvg poisoned | Median clean | Median poisoned |
|---:|---:|---:|---:|---:|
| 42 | 2.29 | 7.39 | 2.82 | 4.75 |
| 43 | 1.58 | 6.51 | 3.17 | 5.46 |
| 44 | 1.94 | 1.76 | 0.70 | 0.53 |
| 45 | 2.99 | 3.70 | 0.53 | 5.28 |
| 46 | 2.99 | 8.80 | 1.94 | 1.94 |
| **Mean** | **2.36** | **5.63** | **1.83** | **3.59** |

The table below gives means across seeds and the mean of the paired
poisoned-minus-clean changes. Accuracy, targeted error, and recall changes are
in percentage points; loss is per-image cross-entropy.

| Aggregation | Metric | Clean mean | Poisoned mean | Paired change |
|---|---|---:|---:|---:|
| FedAvg | Targeted error (%) | 2.36 | 5.63 | +3.27 pp |
| Median | Targeted error (%) | 1.83 | 3.59 | +1.76 pp |
| FedAvg | Accuracy (%) | 79.96 | 78.68 | -1.28 pp |
| Median | Accuracy (%) | 81.58 | 81.10 | -0.47 pp |
| FedAvg | Kidney-left recall (%) | 83.98 | 76.20 | -7.78 pp |
| Median | Kidney-left recall (%) | 83.17 | 81.90 | -1.27 pp |
| FedAvg | Loss | 0.7101 | 0.7508 | +0.0407 |
| Median | Loss | 0.6429 | 0.6641 | +0.0212 |

Median's mean targeted-error increase was smaller in this pilot, but the
effect varies by seed (including a larger increase under median for seed 45).
Clean median accuracy was 1.62 pp higher on average, while clean kidney-left
recall was 0.81 pp lower. These are exploratory observations from five seeds
and a previously used validation set, not evidence of general protection.
Do not use the test split until the broader attack and defense protocols are
fixed.
