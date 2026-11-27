# Label-Flipping Results

This document tracks label-flipping protocol decisions, verification, and
high-level findings. The per-run JSON records are the source of truth for full
metrics, partition manifests, and artifact paths. Clean federated controls are
documented separately in [baseline_results.md](baseline_results.md).

## Completed Protocol-v2: OrganAMNIST Dose Response

**Status:** Complete for seeds 42–51. The primary endpoint and paired
seed-level analysis below are descriptive; no test-set evaluation was used.
The source-of-truth per-run JSON files and model artifacts are under
`generation-v3/protocol-v2`.

### Frozen conditions

This study estimates how targeted label-flip intensity changes kidney-left to
kidney-right validation errors in the existing OrganAMNIST federated setup.
It varies dose for one poisoned client; it is not an attacker-count sweep.

| Setting | Frozen value |
|---------|--------------|
| Dataset and input | OrganAMNIST, 28×28 grayscale |
| Source → target | kidney-left (class 4) → kidney-right (class 5) |
| Clients and aggregation | 20 clients; FedAvg; all clients participate each round |
| Poisoned clients | 1 per nonzero-dose run, selected deterministically from the seed |
| Flip doses | 0%, 25%, 50%, 75%, and 100% of eligible class-4 labels on that client |
| Seeds | 42–51 |
| Training | 5 rounds, 1 local epoch per round, learning rate 0.1, batch size 32 |
| Attack timing | Same label transform in every training round, starting at round 1 |
| Evaluation | Clean validation split during training; test evaluation disabled |

For each seed, all doses use the same seeded partition and poisoned client.
Flipped examples are selected by a seeded ordering, so lower-dose selections
are nested within higher-dose selections. Requested fractions are rounded to
integer example counts and both the request and actual counts are recorded.
The 0% condition is the matched clean control with no poisoned clients.

### Execution and validation

`run-1` contains the ten clean controls and the four nonzero doses for seeds
42–50 (36 poisoned records). `run-2` contains the four nonzero doses for seed
51 and a repeated seed-51 clean control. Thus there are 40 nonzero-dose runs
and ten unique clean controls; retain the duplicate control as a rerun record,
but include seed 51 only once in the analysis.

The 40 poisoned records completed five training rounds with 20 client results
and no client failures per round. Seed-51's five `run-2` records were checked
for completed status, finite recorded metrics, valid-run status, partition
coverage declarations, and existing model/checkpoint references. Their final
model tensors loaded and were finite. The new clean control matches both the
original protocol-v2 seed-51 control and the seed-51 baseline validation
metrics for rounds 0–5. All test metrics are null, as specified.

- Earlier runs: [protocol-v2/run-1 results](results/poisoning/generation-v3/protocol-v2/run-1/organamnist/)
- Seed-51 completion and repeated control: [protocol-v2/run-2 results](results/poisoning/generation-v3/protocol-v2/run-2/organamnist/)
- Seed-51 checkpoints: [protocol-v2/run-2 models](models/poisoning/generation-v3/protocol-v2/run-2/organamnist/)

The original Flower CLI hit a FAB-size limit after result files accumulated in
the app directory. The default FAB inclusion behavior was bundling generated
JSON results along with the app. An explicit source-only `fab-include` in
`pyproject.toml` keeps generated results out of the input bundle; a local build
verified a 100,166-byte FAB with no generated run records. This packaging fix
does not change the result JSON format or storage paths.

The partition manifests record per-client class counts and coverage checks,
not exact example indices or a per-example before/after label audit. The
deterministic transform can be replayed given the same source, seed, and data,
but those records alone do not independently prove the exact selected
examples. The run JSON also does not capture a complete code revision and
dependency/runtime snapshot.

### Outcomes and analysis

**Primary endpoint:** round-5 validation class-4→class-5 error rate, calculated
as true class-4 examples predicted as class 5 divided by the 568 true
class-4 validation examples. Each cell gives the attack error rate and its
paired difference from that seed's 0% control.

| Seed | 25% | 50% | 75% | 100% |
|-----:|----:|----:|----:|-----:|
| 42 | 5.46% (+3.17 pp) | 3.87% (+1.58 pp) | 4.40% (+2.11 pp) | 7.39% (+5.11 pp) |
| 43 | 3.87% (+2.29 pp) | 14.44% (+12.85 pp) | 2.46% (+0.88 pp) | 6.51% (+4.93 pp) |
| 44 | 0.18% (-1.76 pp) | 0.00% (-1.94 pp) | 0.35% (-1.58 pp) | 1.76% (-0.18 pp) |
| 45 | 0.88% (-2.11 pp) | 1.94% (-1.06 pp) | 1.41% (-1.58 pp) | 3.70% (+0.70 pp) |
| 46 | 1.76% (-1.23 pp) | 2.29% (-0.70 pp) | 3.52% (+0.53 pp) | 8.80% (+5.81 pp) |
| 47 | 7.39% (-2.99 pp) | 12.32% (+1.94 pp) | 16.90% (+6.51 pp) | 10.92% (+0.53 pp) |
| 48 | 2.46% (+0.00 pp) | 1.94% (-0.53 pp) | 3.17% (+0.70 pp) | 2.46% (+0.00 pp) |
| 49 | 7.92% (-1.94 pp) | 7.75% (-2.11 pp) | 8.27% (-1.58 pp) | 9.33% (-0.53 pp) |
| 50 | 0.70% (-1.76 pp) | 3.17% (+0.70 pp) | 3.70% (+1.23 pp) | 2.64% (+0.18 pp) |
| 51 | 2.64% (+0.88 pp) | 1.94% (+0.18 pp) | 3.70% (+1.94 pp) | 2.29% (+0.53 pp) |

Across seeds, actual flipped-label counts ranged from 48–52 at 25%, 96–104
at 50%, 143–157 at 75%, and 191–209 at 100%. The JSON records retain each
seed's exact recorded count, client ID, and full round metrics.

Paired changes are attack minus clean control. The intervals are two-sided
95% Student's t intervals over the ten seed-level changes; they are
descriptive, not significance tests.

| Dose | Targeted error change (pp) | Accuracy change (pp) | Loss change | kidney-left recall change (pp) |
|-----:|---------------------------:|---------------------:|------------:|-------------------------------:|
| 25% | -0.55 ± 2.05 [-2.02, +0.92] | -0.20 ± 2.82 [-2.21, +1.82] | -0.0012 ± 0.0852 [-0.0621, +0.0598] | +0.21 ± 8.75 [-6.04, +6.47] |
| 50% | +1.09 ± 4.35 [-2.02, +4.20] | -4.05 ± 11.12 [-12.00, +3.90] | +0.1265 ± 0.4209 [-0.1746, +0.4277] | -8.27 ± 14.28 [-18.49, +1.94] |
| 75% | +0.92 ± 2.42 [-0.82, +2.65] | -0.93 ± 5.17 [-4.62, +2.77] | +0.0056 ± 0.1341 [-0.0903, +0.1016] | -5.48 ± 5.66 [-9.53, -1.42] |
| 100% | +1.71 ± 2.50 [-0.08, +3.50] | +0.26 ± 6.87 [-4.66, +5.17] | -0.0049 ± 0.1389 [-0.1043, +0.0945] | -7.36 ± 7.39 [-12.64, -2.08] |

Cells report mean ± standard deviation [95% CI]. The primary endpoint varies
substantially by seed and does not show a consistent dose-response pattern;
all four targeted-error intervals include zero. These exploratory results do
not establish a uniform attack effect. Validation data remained clean. Test
data must remain unused for attack selection and be evaluated only after
analysis is locked.

### Run validity and deviations

A valid run completes all five rounds with 20 client results and no client
failures per round, finite metrics and model parameters, and a partition
manifest asserting that every training example is assigned exactly once.
Only selected source-class labels on selected poisoned clients may change,
and they must change to class 5. Preserve failed or incomplete records and
document reruns rather than silently excluding conditions. Changes to
dataset, seeds, training settings, client selection, attack mapping, or doses
after execution starts require a new protocol version and a documented
deviation.

## Frozen Protocol-v3: OrganAMNIST Attacker Count × Flip Dose

**Status:** The frozen matrix completed execution, but verification found
three invalid cells that reproduced on rerun. The factorial analysis is not
ready to finalize. This study is separate from the completed single-attacker
protocol-v2 and estimates how attacker count and per-attacker flip dose,
including their interaction, change kidney-left-to-kidney-right validation
errors.

| Setting | Frozen value |
|---------|--------------|
| Dataset and input | OrganAMNIST, 28×28 grayscale |
| Source → target | kidney-left (class 4) → kidney-right (class 5) |
| Clients and aggregation | 20 clients; FedAvg; all clients participate each round |
| Poisoned clients | 1, 2, 4, or 8 per attack run (5%, 10%, 20%, or 40% of clients) |
| Flip doses | 25%, 50%, 75%, or 100% of eligible class-4 labels on each poisoned client |
| Seeds | 42–51 |
| Training | 5 rounds, 1 local epoch per round, learning rate 0.1, batch size 32 |
| Attack timing | Same seeded label transform in every training round, starting at round 1 |
| Evaluation | Clean validation split during training; test evaluation disabled |
| Output location | `results/poisoning/generation-v3/protocol-v3/run-N/organamnist/` |

From `quickstart-pytorch/`, run the frozen matrix with:

```bash
PATH=".venv/bin:$PATH" \
SEEDS="42 43 44 45 46 47 48 49 50 51" \
POISONED_CLIENT_COUNTS="1 2 4 8" \
POISON_FRACTIONS="0.25 0.5 0.75 1.0" \
GENERATION_ID="generation-v3" \
PROTOCOL_ID="protocol-v3" \
NUM_CLIENTS=20 \
NUM_SERVER_ROUNDS=5 \
LOCAL_EPOCHS=1 \
LEARNING_RATE=0.1 \
BATCH_SIZE=32 \
bash ./run_label_flipping_experiments.sh organamnist 4 5
```

The flip fraction is **per poisoned client**, not the fraction of all source
labels across the federation. Increasing attacker count therefore increases
the total number of potentially flipped training labels. Cross every attacker
count with every nonzero dose (16 attack conditions per seed). Include one
zero-poisoned-client, zero-dose clean control per seed. The current runner
records that control once for each seed in an execution, so a full ten-seed
execution contains 170 runs (160 attack conditions plus 10 controls).

### Execution and validity

The initial execution, [run-1 results](results/poisoning/generation-v3/protocol-v3/run-1/organamnist/)
and [run-1 models](models/poisoning/generation-v3/protocol-v3/run-1/organamnist/),
contains all 170 unique seed-by-condition cells. All runs completed five
rounds with 20 client results and no client failures per round. Partition
coverage checks passed, recorded flip counts match the requested per-client
fractions, and test metrics are null. Finite-value checks passed for 167
cells. Three cells have non-finite training/validation losses and model
parameters:

| Seed | Poisoned clients | Flip fraction | First non-finite round |
|-----:|-----------------:|--------------:|-----------------------:|
| 47 | 4 | 50% | 2 |
| 48 | 4 | 25% | 2 |
| 50 | 4 | 50% | 4 |

These were rerun with unchanged settings in
[run-2 results](results/poisoning/generation-v3/protocol-v3/run-2/organamnist/)
and [run-2 models](models/poisoning/generation-v3/protocol-v3/run-2/organamnist/).
Each rerun completed all five rounds with no client failures, but reproduced
the same non-finite losses and parameters at the same rounds. The run-2 records
mark `metrics_finite`, `model_parameters_finite`, and `valid_run` as false.
The original run-1 records predate the finite-value validity check and
incorrectly mark these runs valid. The check is now enforced when new records
are written; see `pytorchexample/experiments/shared/server.py`.

- Seed 47, 4 clients, 50%: [run-1 record](results/poisoning/generation-v3/protocol-v3/run-1/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p500_s4_t5/seed-47_run-747155533925153753.json) · [run-2 rerun](results/poisoning/generation-v3/protocol-v3/run-2/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p500_s4_t5/seed-47_run-10086339224390307204.json)
- Seed 48, 4 clients, 25%: [run-1 record](results/poisoning/generation-v3/protocol-v3/run-1/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p250_s4_t5/seed-48_run-13175489404516207263.json) · [run-2 rerun](results/poisoning/generation-v3/protocol-v3/run-2/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p250_s4_t5/seed-48_run-7225105320867385753.json)
- Seed 50, 4 clients, 50%: [run-1 record](results/poisoning/generation-v3/protocol-v3/run-1/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p500_s4_t5/seed-50_run-18189058754778482809.json) · [run-2 rerun](results/poisoning/generation-v3/protocol-v3/run-2/organamnist/label_flip_r5_e1_lr0p100_clients4_f0p500_s4_t5/seed-50_run-4125338092416594049.json)

The matrix therefore has 170 unique cells but only 167 valid cells; the three
reruns are repeat attempts, not additional seed observations. Preserve both
attempts and exclude the affected cells from primary analysis until the
numerical instability is understood and a protocol deviation or new protocol
version is approved. The valid OrganAMNIST records are included in the
cross-dataset summary below; the three affected cells remain excluded.

### Offline diagnosis of the non-finite updates

A CPU replay from the saved pre-failure global checkpoints localizes the
divergence to client-local SGD, not the validation or result-serialization
path. For seed 47 (4 clients, 50%), poisoned client 5 first produced a
non-finite loss at batch 50 of round 2; its preceding finite batch losses had
already grown to `2.8e19`. For seed 48 (4 clients, 25%), unpoisoned client 13
first produced a non-finite loss at batch 41 of round 2, after a finite loss
of `1.9e28`. In both cases the round-1 global checkpoint was finite.

Seed 50 (4 clients, 50%) shows the escalation earlier: the unpoisoned client 9
reached a local parameter magnitude of about 410 in round 2, and unpoisoned
client 19 reached about 2,945 in round 3. The corresponding saved global
checkpoints grew from maximum absolute parameter values of about 20.35 to
147.19. A replay of those two rounds reproduced the saved aggregate tensors
exactly. In round 4, unpoisoned client 9's loss rose above `1e14` before
becoming non-finite at batch 27.

Two failures therefore first appear in unpoisoned clients. The evidence points
to unstable local optimization under the frozen learning rate (`0.1`) and
momentum (`0.9`); it does not establish that the label-flip transform itself
is faulty. These are offline replays, not per-client traces captured during
the original Flower runs. Keep protocol-v3 settings and results unchanged;
any mitigation such as changing the learning rate or adding gradient clipping
requires an explicitly versioned protocol decision.

Use the same seed-specific data partition in every condition. The existing
seeded selection selects nested poisoned-client sets as count increases and
nested flipped-example sets as dose increases; keep this behavior unchanged.
Record actual flipped-label counts per client. Compare each condition to its
same-seed clean control. Do not count duplicate clean-control reruns as
additional seeds.

The primary endpoint is the round-5 validation class-4→class-5 error rate.
For each count-by-dose condition, report all ten seed-level rates and paired
changes from clean, plus the mean, standard deviation, and two-sided 95%
Student's t interval across seed-level changes. Report class-4 recall,
overall validation accuracy, per-image cross-entropy, confusion matrices,
and actual flipped-label counts as secondary outcomes. Treat intervals as
descriptive; do not treat validation images as independent replicates or make
significance claims. Keep test evaluation disabled until the protocol and
analysis are locked.

Apply the protocol-v2 run-validity criteria above. Preserve per-run JSON,
partition manifests, plots, and checkpoints under a new execution ID. Since
manifests currently store class counts rather than exact example indices,
verify deterministic partition and label-transform behavior against the
same code and dataset before interpreting the factorial comparisons. Any
change to these frozen conditions requires another protocol version or a
documented deviation before execution.

## OrganAMNIST: kidney-left to kidney-right, seeds 42–46

**Status:** Completed five-seed exploratory trial. Validation was used
throughout training, and test evaluation was disabled.

| Setting | Value |
|---------|-------|
| Dataset | OrganAMNIST, 28×28 |
| Source → target | kidney-left (ID 4) → kidney-right (ID 5) |
| Clients | 20 |
| Poisoned clients | 1 (5%); one seeded-random client per run |
| Flip fraction | 100% of each selected client's eligible kidney-left training labels |
| Timing | Every training round, starting at round 1 |
| Training | 5 rounds, 1 local epoch, learning rate 0.1, batch size 32, FedAvg |
| Seeds | 42, 43, 44, 45, 46 |

The seed-42 poisoned client had 199 kidney-left examples, and all 199 labels
were flipped (about 0.58% of the 34,561-example training split). Validation
labels remained clean.

### Round-5 validation comparisons

Each clean control uses the same seed and client partition as its saved
baseline; per-round validation metrics match exactly.

| Seed | Flipped labels | Accuracy, clean → attack | Loss, clean → attack | kidney-left recall, clean → attack | kidney-left → kidney-right, clean → attack |
|-----:|---------------:|-------------------------:|--------------------:|-----------------------------------:|-----------------------------------------:|
| 42 | 199/199 | 81.11% → 87.18% | 0.6764 → 0.5150 | 87.32% → 84.33% | 2.29% (13/568) → 7.39% (42/568) |
| 43 | 201/201 | 76.80% → 76.06% | 0.7774 → 0.8219 | 81.87% → 67.25% | 1.58% (9/568) → 6.51% (37/568) |
| 44 | 191/191 | 80.30% → 64.38% | 0.7687 → 1.0792 | 82.39% → 63.91% | 1.94% (11/568) → 1.76% (10/568) |
| 45 | 195/195 | 78.89% → 83.18% | 0.6863 → 0.6558 | 84.86% → 83.80% | 2.99% (17/568) → 3.70% (21/568) |
| 46 | 195/195 | 82.70% → 82.61% | 0.6417 → 0.6821 | 83.45% → 81.69% | 2.99% (17/568) → 8.80% (50/568) |

The target-directed error rate was 0% through round 3, 0.70% (4/568) at round
4, and 7.39% (42/568) at round 5 for seed 42. Across the five paired seeds,
the mean change (attack minus clean) was −1.28 percentage points in accuracy,
+0.0407 in loss, −7.78 pp in kidney-left recall, and +3.27 pp in targeted
error rate. Targeted error increased in four pairs and decreased slightly in
seed 44; overall accuracy varied, including a substantial decrease in seed 44.
The results show a variable class-specific effect and do not establish a
uniform effect across seeds.

### Label-transformation verification

The seed-42 run record reports client 2 and 199 flipped examples. A replay
using the same OrganAMNIST training split, seed-42 client partition, and
deterministic transform confirmed that all 199 kidney-left labels in that
client's 1,728 examples became kidney-right, no kidney-left labels remained in
that client, and every non-source label was unchanged. The focused data tests
also passed with:

```bash
python -m unittest discover -s tests -p 'test_data.py'
```

The original run did not persist a per-example before/after label audit. The
verification reproduces the recorded transform against the exact deterministic
partition; it is not a saved snapshot of the tensors used during the original
Flower execution.

### Run records

- Poisoned run:
  [seed 42 JSON](results/poisoning/generation-v3/protocol-v1/run-1/organamnist/label_flip_r5_e1_lr0p100_clients1_f1p000_s4_t5/seed-42_run-13042088134346131756.json)
- Matched clean control:
  [seed 42 JSON](results/poisoning/generation-v3/protocol-v1/run-1/organamnist/label_flip_r5_e1_lr0p100_clients0_f0p000_s4_t5/seed-42_run-12640462282327131058.json)
- Saved clean baseline:
  [seed 42 JSON](results/baseline/generation-v3/protocol-v1/run-4/organamnist/baseline_r5_e1_lr0p100/seed-42_run-13016821332549798429.json)
- Seeds 43–46:
  [clean controls](results/poisoning/generation-v3/protocol-v1/run-2/organamnist/label_flip_r5_e1_lr0p100_clients0_f0p000_s4_t5/) · [poisoned runs](results/poisoning/generation-v3/protocol-v1/run-2/organamnist/label_flip_r5_e1_lr0p100_clients1_f1p000_s4_t5/)

All five poisoned runs and their clean controls completed five rounds with 20
client results per round and zero client failures. The poisoned records include
their confusion matrices, plots, checkpoint references, and partition
manifests.

## OrganCMNIST and OrganSMNIST: seeds 42–46

The same one-client, 100%-flip condition was completed for five seeds on both
datasets. Their selected clean-baseline settings use 3 local epochs (the other
settings match the OrganAMNIST trial). The kidney class IDs are 4 → 5 in both
datasets. Each clean control matches its saved seed-specific federated
baseline's per-round validation metrics and partition.

The table reports each seed's round-5 validation metrics. Source-to-target
error is the fraction of true kidney-left validation examples predicted as
kidney-right; each row is paired with its same-seed clean control.

| Dataset | Seed | Flipped labels | Accuracy, clean → attack | Loss, clean → attack | kidney-left recall, clean → attack | kidney-left → kidney-right, clean → attack |
|---------|-----:|---------------:|-------------------------:|--------------------:|-----------------------------------:|-----------------------------------------:|
| OrganCMNIST | 42 | 52/52 | 89.76% → 90.26% | 0.4381 → 0.4288 | 95.45% → 98.48% | 2.27% (3/132) → 0% (0/132) |
| OrganCMNIST | 43 | 49/49 | 85.41% → 86.66% | 0.5653 → 0.5449 | 100.00% → 96.21% | 0% (0/132) → 2.27% (3/132) |
| OrganCMNIST | 44 | 43/43 | 87.58% → 87.25% | 0.5064 → 0.4544 | 97.73% → 93.94% | 0% (0/132) → 3.03% (4/132) |
| OrganCMNIST | 45 | 41/41 | 86.62% → 87.29% | 0.5059 → 0.5437 | 88.64% → 92.42% | 2.27% (3/132) → 2.27% (3/132) |
| OrganCMNIST | 46 | 55/55 | 88.67% → 85.28% | 0.4743 → 0.5833 | 92.42% → 90.15% | 4.55% (6/132) → 2.27% (3/132) |
| OrganSMNIST | 42 | 55/55 | 67.50% → 71.78% | 0.9673 → 0.9419 | 82.14% → 24.29% | 2.86% (4/140) → 59.29% (83/140) |
| OrganSMNIST | 43 | 66/66 | 67.25% → 70.43% | 1.0183 → 0.8522 | 57.86% → 0% | 30.71% (43/140) → 90.71% (127/140) |
| OrganSMNIST | 44 | 47/47 | 75.20% → 75.77% | 0.6920 → 0.6579 | 7.14% → 1.43% | 86.43% (121/140) → 93.57% (131/140) |
| OrganSMNIST | 45 | 60/60 | 69.74% → 70.27% | 0.9386 → 0.8904 | 30.71% → 7.86% | 55.71% (78/140) → 77.14% (108/140) |
| OrganSMNIST | 46 | 64/64 | 72.72% → 74.92% | 0.7941 → 0.7615 | 12.86% → 2.86% | 70.00% (98/140) → 80.00% (112/140) |

Across the five paired seeds, the mean change (attack minus clean) for
OrganCMNIST was −0.26 percentage points in accuracy, +0.0130 in loss,
−0.61 pp in kidney-left recall, and +0.15 pp in targeted error rate. The
targeted-error direction was inconsistent across seeds, so this condition did
not show a replicated OrganCMNIST attack effect.

For OrganSMNIST, the corresponding mean paired changes were +2.15 pp in
accuracy, −0.0613 in loss, −30.86 pp in kidney-left recall, and +31.00 pp in
targeted error rate. Targeted error increased in all five pairs, while
kidney-left recall decreased in all five. The higher overall accuracy in each
poisoned run does not contradict these class-specific effects and is not
evidence of a general benefit. Both findings remain limited to this
one-client, full-flip condition and these five seeds.

All ten poisoned records and their controls completed five rounds with 20
results and zero client failures per round; test evaluation was disabled.
For each seed, the recorded number of flipped labels equals the selected
client's kidney-left count in its saved partition manifest (240 labels total
for OrganCMNIST; 292 for OrganSMNIST). The run records do not persist a
per-example before/after label audit. A deterministic replay against the
cached official training splits for seeds 42–46 reproduced the selected client
and flip count for both datasets. It confirmed that every source label became
kidney-right, no source labels remained on the selected client, and no
non-source labels changed. The replay was also run for all five OrganAMNIST
seeds. It reproduces the transforms from the saved configuration and matching
training data, not a snapshot of the tensors used during execution.

Run records:

- OrganCMNIST seed 42: [clean control](results/poisoning/generation-v3/protocol-v1/run-3/organcmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/seed-42_run-17716134690215662785.json) · [poisoned run](results/poisoning/generation-v3/protocol-v1/run-3/organcmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/seed-42_run-13627381287354949253.json)
- OrganCMNIST seeds 43–46: [clean controls](results/poisoning/generation-v3/protocol-v1/run-5/organcmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/) · [poisoned runs](results/poisoning/generation-v3/protocol-v1/run-5/organcmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/)
- OrganSMNIST seed 42: [clean control](results/poisoning/generation-v3/protocol-v1/run-4/organsmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/seed-42_run-15574750315785558871.json) · [poisoned run](results/poisoning/generation-v3/protocol-v1/run-4/organsmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/seed-42_run-16040722783643213093.json)
- OrganSMNIST seeds 43–46: [clean controls](results/poisoning/generation-v3/protocol-v1/run-6/organsmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/) · [poisoned runs](results/poisoning/generation-v3/protocol-v1/run-6/organsmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/)

## OrganCMNIST and OrganSMNIST stability pilot (protocol-v4)

**Status:** All 12 pilot runs passed validity checks: three targeted attack
conditions and their same-seed clean controls on each dataset. The attacks
reused the OrganAMNIST failure configurations (seed 47, 4 clients, 50%; seed
48, 4 clients, 25%; seed 50, 4 clients, 50%) with the same class-4→class-5
target, 20 clients, 5 rounds, 1 local epoch, learning rate 0.1, and batch size
32. Test evaluation remained disabled.

All runs completed five rounds with 20 client results and zero client failures
per round. Recorded metrics, final model parameters, and all 60 round
checkpoints were finite; all 12 final models and 60 checkpoints loaded
successfully. Partition manifests passed their coverage checks, matched
between each clean/attack pair, and the recorded flip counts agreed with the
selected clients' source-class counts. These are stability checks, not
full-matrix effect estimates. Artifacts are under
`results/poisoning/generation-v3/protocol-v4/run-*/` and
`models/poisoning/generation-v3/protocol-v4/run-*/`.

The pilot supports proceeding to full OrganCMNIST and OrganSMNIST sweeps under
the same training settings. Keep the pilot-only `protocol-v4` artifacts
separate; record and execute the full matrices under a new protocol ID.

## Verification suite

The full quickstart-pytorch suite passed (38 tests):

```bash
cd quickstart-pytorch
.venv/bin/python -m unittest discover -s tests -v
```

The regression checks cover the baseline Flower client/server clean defaults,
deterministic full label flipping for OrganCMNIST and OrganSMNIST, loading a
saved round checkpoint into the model, majority-class accuracy reference, and
overfitting a small image subset with the shared CNN training loop. In
addition, all 90 saved final and per-round checkpoints from the 15 poisoned
runs loaded strictly into the CNN and produced finite 11-class outputs.

## Refinement notes

- Keep each poisoned run paired with a clean control using the same seed,
  client count, partition, and training settings.
- Change one attack factor at a time. Track targeted source-to-target
  misclassification, source-class recall, overall validation metrics, and
  label-transformation counts.
- Keep test evaluation disabled during refinement. Evaluate test data only
  after the protocol is fixed, and report it separately from validation.
- For every future result, record the generation/protocol/execution IDs, run
  JSON path, clean-control pairing, selected client IDs, flipped-example
  counts, label-audit outcome, and test-evaluation status.

## Frozen full-matrix results: OrganAMNIST, OrganCMNIST, and OrganSMNIST

These result directories are the frozen label-flipping baselines for the
upcoming defense experiments. Each matrix has 16 poisoned conditions (1, 2, 4,
or 8 poisoned clients crossed with per-client flip fractions of 25%, 50%, 75%,
or 100%) and one matched clean control for each seed, 42–51. The common
training settings are 20 clients with FedAvg, 5 rounds, 1 local epoch per
round, learning rate 0.1, and batch size 32. The targeted transform maps
kidney-left (class 4) to kidney-right (class 5) during every training round.
Validation evaluation was enabled; test evaluation was disabled.

| Dataset | Frozen result records | Model artifacts |
|---------|-----------------------|-----------------|
| OrganAMNIST, protocol-v3 | [run-1 results](results/poisoning/generation-v3/protocol-v3/run-1/organamnist/); three failure reruns in [run-2](results/poisoning/generation-v3/protocol-v3/run-2/organamnist/) | [run-1 models](models/poisoning/generation-v3/protocol-v3/run-1/organamnist/) |
| OrganCMNIST, protocol-v5 | [run-4 results](results/poisoning/generation-v3/protocol-v5/run-4/organcmnist/) and [run-5 results](results/poisoning/generation-v3/protocol-v5/run-5/organcmnist/) | [run-4 models](models/poisoning/generation-v3/protocol-v5/run-4/organcmnist/) and [run-5 models](models/poisoning/generation-v3/protocol-v5/run-5/organcmnist/) |
| OrganSMNIST, protocol-v5 | [run-6 results](results/poisoning/generation-v3/protocol-v5/run-6/organsmnist/) | [run-6 models](models/poisoning/generation-v3/protocol-v5/run-6/organsmnist/) |

OrganAMNIST run-1 has 170 result records, but three contain non-finite
metrics/model parameters: seed 47 (4 clients, 50%), seed 48 (4 clients, 25%),
and seed 50 (4 clients, 50%). The three run-2 reruns reproduce those failures;
they are not additional observations. The primary analysis below uses the 167
valid run-1 records and excludes those three cells. For OrganCMNIST, runs 4 and
5 contain the complete 170-cell matrix. The 53 earlier records in runs 1–3 are
partial attempts whose cells all also occur in runs 4–5; do not count them as
extra observations. OrganSMNIST run-6 contains the complete 170-cell matrix.

For each canonical matrix, all expected result and partition JSON files are
present, and final-model and round-checkpoint references resolve. All
OrganCMNIST and OrganSMNIST records report completed, valid runs, finite
metrics and parameters, 20 client results and zero client failures in each of
five rounds. Their validation metrics cover rounds 0–5. OrganAMNIST has the
three numerical failures noted above. Test metrics are null throughout.

### Round-5 validation analysis

The primary endpoint is the targeted class-4→class-5 error rate: true
kidney-left validation examples predicted as kidney-right. Clean and poisoned
rates below are averages over the available seed/control and
condition/seed records, respectively. Paired deltas compare each poisoned
condition with its same-seed clean control. For the intervals, condition-level
deltas are first averaged within each seed, then a two-sided 95% Student's
*t* interval is calculated across the ten seed-level averages. OrganAMNIST
seed averages exclude its affected cell where applicable (15 valid attack
conditions for seeds 47, 48, and 50; 16 for the other seeds). Accuracy and
macro-F1 use the same seed-paired calculation; macro-F1 is the unweighted
mean over the 11 classes. Intervals are descriptive, not significance tests.

| Dataset | Targeted error, clean → poisoned | Targeted-error Δ, pp (95% interval) | Accuracy Δ, pp (95% interval) | Macro-F1 Δ, pp (95% interval) |
|---------|-----------------------------------:|-----------------------------------:|------------------------------:|------------------------------:|
| OrganAMNIST | 3.87% → 13.31% | +9.47 (+6.85, +12.09) | -1.60 (-3.74, +0.54) | -2.50 (-5.86, +0.86) |
| OrganCMNIST | 3.56% → 11.84% | +8.28 (+5.16, +11.41) | -0.97 (-2.16, +0.21) | -2.65 (-4.37, -0.92) |
| OrganSMNIST | 11.07% → 16.32% | +5.25 (+1.60, +8.90) | +0.06 (-1.11, +1.22) | -0.38 (-1.97, +1.22) |

The mean and interval do not show the full spread. Across the ten
seed-averaged paired changes, and across individual condition-by-seed pairs,
the targeted-error changes vary as follows:

| Dataset | Seed-averaged Δ, mean (SD), range (pp) | Positive condition-seed changes |
|---------|----------------------------------------:|---------------------------------:|
| OrganAMNIST | +9.47 (3.67), +4.43 to +15.47 | 121/157 |
| OrganCMNIST | +8.28 (4.37), +2.41 to +15.15 | 130/160 |
| OrganSMNIST | +5.25 (5.10), 0.00 to +16.38 | 96/160 |

The seed-level SD and range summarize variability after averaging the valid
attack conditions within each seed. The positive counts instead describe
individual condition-seed pairs relative to that seed's clean control; they
are not independent replicates. OrganSMNIST therefore has a positive average
but a smaller share of cells that increase the target error.

### Source-class support and confusion slice

The following is the round-5 validation confusion-matrix row for true
kidney-left examples. Support is the number of true kidney-left examples in
each run; prediction percentages are averaged over clean controls or valid
poisoned records. “Other” combines predictions for the remaining nine
classes.

| Dataset | Source support per run | Clean: predicted kidney-left / kidney-right / other | Poisoned: predicted kidney-left / kidney-right / other |
|---------|-----------------------:|----------------------------------------------------:|-------------------------------------------------------:|
| OrganAMNIST | 568 | 81.11% / 3.87% / 15.02% | 61.70% / 13.31% / 24.98% |
| OrganCMNIST | 132 | 29.32% / 3.56% / 67.12% | 15.92% / 11.84% / 72.23% |
| OrganSMNIST | 140 | 4.29% / 11.07% / 84.64% | 1.25% / 16.32% / 82.43% |

This row makes clear that the targeted error is only one part of source-class
performance. In particular, clean OrganCMNIST and OrganSMNIST models already
have low kidney-left recall, and most source-class errors on those datasets
are predictions outside the target class.

The targeted error increases across the three datasets. The condition-averaged
paired increase also rises with poisoned-client count (1, 2, 4, then 8):
OrganAMNIST +0.79, +2.90, +9.16, and +24.96 pp; OrganCMNIST +1.00, +4.41,
+8.33, and +19.38 pp; OrganSMNIST +0.30, +2.61, +6.29, and +11.80 pp.
These summaries average over the four flip fractions and are descriptive.
Overall-accuracy intervals include zero for all three datasets; the
macro-F1 interval includes zero for OrganAMNIST and OrganSMNIST, but not
OrganCMNIST. OrganSMNIST's clean baseline is comparatively weak (45.12%
accuracy and 28.11% macro-F1), so cross-dataset differences in attack effect
should be interpreted in that context.

OrganAMNIST uses protocol-v3, whereas OrganCMNIST and OrganSMNIST use
protocol-v5. Their recorded training settings and attack factors match, but
cross-dataset comparisons are descriptive rather than a single
same-protocol estimate. The successful cells have not all been repeated with
the exact same seed/configuration; these results support cross-seed and
cross-dataset consistency, not bit-for-bit run repeatability. Keep these
result and model directories unchanged as the baseline for defense work.
Continue to use validation for development and reserve test evaluation until
the defense protocol and analysis are locked.
