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

## Verification after code reorganization

The full existing test suite passed all 52 tests on the reorganized code,
including the 20-client even median and rejection of missing, duplicate,
incompatible, and non-finite replies. The tests also check the odd-client
median. No test split was evaluated.

The source-hash audit compared the original pilot manifest with the current
source at commit `722331f535c5ed485ffec662a5d9a55459b6b736`:

- 14 recorded files still at their original paths have matching SHA-256 hashes.
- The label-flipping module and both active Flower dispatchers moved to new
  paths; all three match their original recorded hashes byte-for-byte.
- The shared client and server modules and `pyproject.toml` differ. Their
  refactoring changed imports, strategy placement, dispatch and output-layout
  configuration, so the manifest hashes cannot establish that the current
  implementation is identical.
- Four former baseline/poisoning app wrappers were removed. Their versions at
  the pilot's recorded starting Git revision match the manifest hashes. The
  old pilot script was untracked at the time and has since been replaced by
  `run_median_defense_pilot.sh`; its historical hash cannot be checked against
  a Git-tracked copy. The pilot also used uncommitted changes to the server and
  configuration, so its recorded Git revision alone is not the exact source
  snapshot.

A separate, validation-only seed-42 rerun used the clean committed working
tree at `722331f535c5ed485ffec662a5d9a55459b6b736`. Its four matched clean
and poisoned FedAvg and median runs are in
[`run-2/`](results/defense/generation-v1/protocol-v1/run-2/), with corresponding
models under [`run-2/`](models/defense/generation-v1/protocol-v1/run-2/).
All four completed with 20 successful replies in every round, verified
partitions, finite metrics and weights, and test evaluation disabled. For each
condition, its partition, configuration, per-round validation and targeted
metrics, and all five round-checkpoint tensors matched the original seed-42
pilot exactly.

The recorded Python, platform, CPU/CUDA status, thread settings, dataset
checksum, and dependency versions matched; the editable project dependency
reported the newer Git revision. This one-seed behavioral check supports
continuity after refactoring but does not prove equivalence across all seeds
or environments. The rerun is a repeat observation, not an additional seed
in the pilot summary.

## Stronger attack comparison: four clients, 100% flip

The exploratory comparison uses the completed four-client, 100%-flip median
attacks from
[`protocol-v2/run-1/`](results/defense/generation-v1/protocol-v2/run-1/) and
the historical, validated FedAvg four-client attack and clean records from
[`generation-v3/protocol-v3/run-1/`](results/poisoning/flipping/generation-v3/protocol-v3/run-1/organamnist/).
The median clean controls come from protocol-v1. All attack records used 20
clients, five rounds, one local epoch, learning rate 0.1, and batch size 32.
For each seed, the source partitions and selected four poisoned client IDs
match between the FedAvg and median attacks. All records used here completed
with finite validation metrics, 20 successful replies and no client failures
in every round. Test evaluation remained disabled.

Round-5 validation changes are reported per seed. `Clean difference` is
median-clean minus FedAvg-clean; each `Attack change` is poisoned minus clean
within that aggregation; `Attack-effect difference` is median attack change
minus FedAvg attack change. Rates are percentage points (pp); loss is the
per-image cross-entropy difference. For targeted error and loss, increases
are worse; for recall and accuracy, decreases are worse. Thus, for example,
a negative targeted-error effect difference or a positive recall effect
difference favors median for that metric.

Median-clean seed 42 exists in two valid runs; their round-5 metrics are
identical, and run-2 is used here. Seed 49 has a valid median attack but no
completed median-clean control, so its median paired changes and all 9-seed
paired means exclude it. Its FedAvg change is shown for completeness. The
median attack's unpaired seed-49 round-5 values were 11.44% targeted error,
84.15% kidney-left recall, 85.90% overall accuracy, and 0.5972 loss.

### Targeted error

| Seed | Clean difference | FedAvg attack change | Median attack change | Attack-effect difference |
|---:|---:|---:|---:|---:|
| 42 | +0.53 pp | +26.76 pp | +6.87 pp | -19.89 pp |
| 43 | +1.58 pp | +39.44 pp | +9.86 pp | -29.58 pp |
| 44 | -1.23 pp | +8.27 pp | +1.58 pp | -6.69 pp |
| 45 | -2.46 pp | +33.27 pp | +6.16 pp | -27.11 pp |
| 46 | -1.06 pp | +27.82 pp | +2.64 pp | -25.18 pp |
| 47 | +3.70 pp | +25.70 pp | +7.04 pp | -18.66 pp |
| 48 | +0.70 pp | +15.32 pp | +3.87 pp | -11.44 pp |
| 49 | N/A | +19.89 pp | N/A | N/A |
| 50 | +0.00 pp | +21.30 pp | +2.46 pp | -18.84 pp |
| 51 | +1.23 pp | +11.44 pp | +4.93 pp | -6.51 pp |
| **Mean (9 paired seeds)** | **+0.33 pp** | **+23.26 pp** | **+5.05 pp** | **-18.21 pp** |

### Kidney-left recall

| Seed | Clean difference | FedAvg attack change | Median attack change | Attack-effect difference |
|---:|---:|---:|---:|---:|
| 42 | -6.16 pp | -59.86 pp | -12.32 pp | +47.54 pp |
| 43 | +8.98 pp | -70.60 pp | -26.94 pp | +43.66 pp |
| 44 | +4.93 pp | -48.77 pp | -11.80 pp | +36.97 pp |
| 45 | -9.33 pp | -67.78 pp | +5.28 pp | +73.06 pp |
| 46 | -2.46 pp | -42.96 pp | -7.39 pp | +35.56 pp |
| 47 | -2.82 pp | -36.44 pp | -20.25 pp | +16.20 pp |
| 48 | -3.70 pp | -69.19 pp | -31.16 pp | +38.03 pp |
| 49 | N/A | -30.81 pp | N/A | N/A |
| 50 | +5.46 pp | -42.78 pp | -9.68 pp | +33.10 pp |
| 51 | +14.79 pp | -31.16 pp | -16.73 pp | +14.44 pp |
| **Mean (9 paired seeds)** | **+1.08 pp** | **-52.17 pp** | **-14.55 pp** | **+37.62 pp** |

### Overall accuracy

| Seed | Clean difference | FedAvg attack change | Median attack change | Attack-effect difference |
|---:|---:|---:|---:|---:|
| 42 | -1.22 pp | -6.04 pp | +0.85 pp | +6.89 pp |
| 43 | +5.73 pp | -0.40 pp | -2.23 pp | -1.83 pp |
| 44 | +2.53 pp | -9.81 pp | -7.78 pp | +2.03 pp |
| 45 | +2.57 pp | -8.40 pp | +0.23 pp | +8.63 pp |
| 46 | -1.53 pp | +1.26 pp | -0.54 pp | -1.80 pp |
| 47 | -0.32 pp | +0.46 pp | -0.31 pp | -0.77 pp |
| 48 | -1.85 pp | -3.60 pp | -4.42 pp | -0.82 pp |
| 49 | N/A | +0.15 pp | N/A | N/A |
| 50 | +7.60 pp | +1.11 pp | -3.53 pp | -4.64 pp |
| 51 | +2.39 pp | +0.20 pp | -0.82 pp | -1.02 pp |
| **Mean (9 paired seeds)** | **+1.77 pp** | **-2.80 pp** | **-2.06 pp** | **+0.74 pp** |

### Validation loss

| Seed | Clean difference | FedAvg attack change | Median attack change | Attack-effect difference |
|---:|---:|---:|---:|---:|
| 42 | +0.0077 | +0.1319 | +0.0203 | -0.1116 |
| 43 | -0.1296 | -0.0755 | +0.0837 | +0.1592 |
| 44 | -0.1878 | +0.0811 | +0.2413 | +0.1601 |
| 45 | -0.0258 | +0.2165 | +0.0653 | -0.1511 |
| 46 | -0.0006 | -0.0541 | +0.0466 | +0.1007 |
| 47 | -0.0101 | -0.0840 | -0.0799 | +0.0041 |
| 48 | +0.0293 | +0.0405 | +0.0795 | +0.0390 |
| 49 | N/A | -0.1077 | N/A | N/A |
| 50 | -0.2220 | -0.0669 | +0.1338 | +0.2007 |
| 51 | -0.0580 | +0.0083 | +0.0737 | +0.0654 |
| **Mean (9 paired seeds)** | **-0.0663** | **+0.0220** | **+0.0738** | **+0.0518** |

Across the nine matched seeds, median had a smaller average increase in
targeted error and a smaller average kidney-left recall reduction than
FedAvg. The average overall-accuracy changes were similar, while median's
average loss increase was larger. Individual outcomes vary: at seed 45,
median recall increased by 5.28 pp while FedAvg recall fell by 67.78 pp; at
seed 43, median's accuracy drop was larger than FedAvg's. The result supports
an exploratory, metric-specific comparison, not a general robustness claim.

The reused FedAvg records have no source/environment manifest, and median
clean controls for seeds 43–46 predate the current source hashes. The median
clean seed-49 failure is unresolved. Keep the three invalid protocol-v3
four-client 25–50% cells excluded as documented in
[`poisoning_results.md`](poisoning_results.md). Use validation only until the
final matched protocol and source are fixed; evaluate the test split only
after that decision.

## Seed-49 median-clean failure and disposition

The protocol-v1 median-clean attempts for seed 49 repeatedly stopped in round
2 when validation rejected a non-finite `fc1.weight`. Ray also reported that
its metrics-exporter agent could not connect; that warning concerns telemetry,
not the training exception that aborted aggregation. An offline replay from the
saved round-1 median checkpoint localized the divergence to client 10, batch
48, where loss escalated before the update became non-finite. A diagnostic
local update at learning rate 0.01 stayed finite, but this was not a complete
federated rerun and does not establish that the lower rate fixes the issue.

The failure is therefore recorded as an incomplete control, not a valid
median-clean result. Keep the current finite-update and complete-round checks:
rejecting a non-finite client update is the intended fail-closed behavior, and
the evidence points to client-training divergence rather than demonstrating a
bug in the coordinate-wise median calculation. Do not silently drop the
reply, accept the run, or change only seed 49's learning rate. Any adopted
stability change, such as a lower learning rate or gradient clipping, changes
the protocol and should be evaluated in a new protocol with matched clean and
attack conditions for both aggregators across the seed set. The later
protocol-v2 four-client attack for seed 49 completed, but it does not supply
the missing median-clean control. Continue using validation only until the
final protocol is fixed; keep the test split untouched until then.
