
# Flower Federated Learning Baselines

## Previous Reference: Frozen Clean Baseline (Validation-v2)

**Status:** Existing five-seed training runs with post-hoc final-checkpoint metrics added; a corrected rerun is still needed for round-by-round outputs.

**Provenance:** Committer-confirmed (default).

**Use:** The `validation_final_recomputed` records provide corrected final-round validation metrics. Treat the original per-round records as historical output, and do not treat this post-hoc evaluation as a new training sweep.

All 105 saved final model weights were reevaluated on the centralized validation split with per-image cross-entropy, per-class precision/recall/F1/support, and confusion matrices. These metrics are added to each run's JSON under `metrics.validation_final_recomputed`; the stored final-round accuracies match the original records exactly. The original per-round loss values still equally weight batch means, and the original client training metrics are unchanged. Since no intermediate model checkpoints were saved, corrected per-round metrics require rerunning training. Future runs are grouped under a category and a single UTC `protocol_<timestamp>` directory; the next corrected baseline sweep will be written under `results/baseline/` and `models/baseline/` without changing these Validation-v2 results.

### Frozen training and evaluation protocol

- Datasets: OrganAMNIST, OrganCMNIST, and OrganSMNIST, using MedMNIST 28x28 grayscale images.
- Model: the existing 11-class CNN in [`pytorchexample/task.py`](pytorchexample/task.py).
- Framework and aggregation: Flower 1.30.x with FedAvg.
- Data split: MedMNIST `train` is partitioned uniformly among 20 clients; centralized `val` is used at initialization and after every server round.
- Partitioning: for each seed, a seeded random permutation of the training indices is interleaved across clients. Client training uses batch size 32 and seeded shuffling.
- Optimization: cross-entropy loss and SGD with momentum 0.9. The learning rate and local epoch count are set per dataset below.
- Seeds: 42, 43, 44, 45, and 46. The global model and data ordering are seeded for each run.
- Participation: all 20 clients are required to train each round; no client-side evaluation is requested.
- Test set: evaluation disabled for this sweep. All 105 records have `test_final: null`; test metrics were not used to select the settings.
- Run integrity: all 105 records completed, had 20 client results in every round, and recorded zero client failures. Run-specific JSON records are in [`results/baseline/validation-v2/`](results/baseline/validation-v2/).

### Complete sweep summary

Accuracy is the mean final-round validation accuracy across five seeds; “±” is the population standard deviation in percentage points. Mean final loss is recomputed with per-image weighting from the saved final model checkpoints. Per-run confusion matrices and class metrics are in the corresponding JSON records. The selected candidate setting for each dataset is bolded.

| Dataset | Rounds | Local epochs | Learning rate | Final validation accuracy | Mean final loss |
|---------|-------:|-------------:|--------------:|---------------------------:|----------------:|
| OrganAMNIST | 3 | 1 | 0.001 | 16.83% ± 1.82 pp | 2.350 |
| OrganAMNIST | 3 | 1 | 0.01 | 49.01% ± 4.27 pp | 1.379 |
| OrganAMNIST | 3 | 1 | 0.1 | 57.49% ± 4.73 pp | 1.347 |
| OrganAMNIST | 3 | 3 | 0.001 | 30.75% ± 12.13 pp | 2.005 |
| OrganAMNIST | 5 | 1 | 0.001 | 21.50% ± 6.58 pp | 2.281 |
| **OrganAMNIST** | **5** | **1** | **0.1** | **79.96% ± 2.00 pp** | **0.710** |
| OrganAMNIST | 5 | 3 | 0.1 | 70.08% ± 16.51 pp | 1.023 |
| OrganCMNIST | 3 | 1 | 0.001 | 14.43% ± 3.26 pp | 2.388 |
| OrganCMNIST | 3 | 1 | 0.01 | 17.93% ± 0.00 pp | 2.272 |
| OrganCMNIST | 3 | 1 | 0.1 | 50.61% ± 5.94 pp | 1.473 |
| OrganCMNIST | 3 | 3 | 0.001 | 17.93% ± 0.00 pp | 2.355 |
| OrganCMNIST | 5 | 1 | 0.001 | 16.30% ± 2.01 pp | 2.380 |
| OrganCMNIST | 5 | 1 | 0.1 | 73.31% ± 2.99 pp | 0.898 |
| **OrganCMNIST** | **5** | **3** | **0.1** | **87.61% ± 1.52 pp** | **0.498** |
| OrganSMNIST | 3 | 1 | 0.001 | 15.56% ± 5.45 pp | 2.386 |
| OrganSMNIST | 3 | 1 | 0.01 | 20.02% ± 0.00 pp | 2.310 |
| OrganSMNIST | 3 | 1 | 0.1 | 32.41% ± 5.89 pp | 1.855 |
| OrganSMNIST | 3 | 3 | 0.001 | 20.02% ± 0.00 pp | 2.349 |
| OrganSMNIST | 5 | 1 | 0.001 | 17.77% ± 4.54 pp | 2.377 |
| OrganSMNIST | 5 | 1 | 0.1 | 46.16% ± 4.65 pp | 1.358 |
| **OrganSMNIST** | **5** | **3** | **0.1** | **70.48% ± 3.07 pp** | **0.882** |

### Candidate settings to revalidate for poisoning comparisons

| Dataset | Rounds | Local epochs | Learning rate | Seeds |
|---------|-------:|-------------:|--------------:|-------|
| OrganAMNIST | 5 | 1 | 0.1 | 42–46 |
| OrganCMNIST | 5 | 3 | 0.1 | 42–46 |
| OrganSMNIST | 5 | 3 | 0.1 | 42–46 |

These were the best mean validation-accuracy settings among the seven tested configurations for each dataset, not estimates of globally optimal settings. Reconfirm them in the corrected sweep before using them as poisoning controls. Once confirmed, keep each dataset's configuration fixed across its clean and poisoned runs; do not retune settings per attack condition. The shared validation set was used for this selection, so final test evaluation remains reserved until the protocol is locked.

### Why earlier results are not comparable

The earliest commit on this branch (`dea3511`, relative to `origin/dev`) changed server-side round evaluation from the official MedMNIST `test` split to `val`. Before that change, `global_evaluate` used `load_centralized_dataset()`, which loaded `test`; afterward, it explicitly evaluated on `val`. This confirms that the earlier implementation exposed test-set metrics during repeated evaluation. If those metrics informed model or hyperparameter selection, that creates test-set leakage risk. It does **not** show that validation examples were included in client training: client training data was partitioned from `train`.

Earlier results are retained below as project history. Their environments and protocols are not consistently verified, and they should not be combined with, directly compared to, or used to select settings for Validation-v2.

## Historical Results

### Earlier OrganAMNIST baseline

The settings below describe the originally reported OrganAMNIST historical baseline.
- Clients: 20
- Federated rounds: 3
- Local epochs: 1
- Learning rate: 0.001
- Batch size: 32

### Original reported metrics

These are retained for historical context and have not been verified against the current seeded implementation.

| Round | Accuracy | Evaluation Loss |
|-------|----------|-----------------|
| Initial | 10.60% | 2.3914 |
| 1 | 12.85% | 2.3758 |
| 2 | 18.48% | 2.3601 |
| 3 | 18.48% | 2.3400 |

### Three-seed baseline summaries

#### OrganAMNIST, seeds 42–44
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 10.293621% | 0.725681% | 2.396808 | 0.003825 |
| 1 | 14.112949% | 3.812424% | 2.381487 | 0.001020 |
| 2 | 18.474144% | 0.006495% | 2.367219 | 0.003002 |
| 3 | 18.515394% | 0.064951% | 2.352745 | 0.007457 |

Individual records:
- [seed 42](results/baseline/organamnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/baseline/organamnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/baseline/organamnist/baseline_r3_e1_lr0p001/seed-44.json)

#### OrganCMNIST, seeds 42–44
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 10.365953% | 3.115142% | 2.393640 | 0.007745 |
| 1 | 13.222168% | 7.905748% | 2.388110 | 0.007652 |
| 2 | 16.869523% | 6.838005% | 2.382765 | 0.007564 |
| 3 | 19.855566% | 4.293586% | 2.377588 | 0.007500 |

Individual records:
- [seed 42](results/baseline/organcmnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/baseline/organcmnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/baseline/organcmnist/baseline_r3_e1_lr0p001/seed-44.json)

#### OrganSMNIST, seeds 42–44
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 9.848571% | 2.792570% | 2.390166 | 0.014532 |
| 1 | 14.905026% | 7.964018% | 2.382573 | 0.013649 |
| 2 | 18.360334% | 7.333288% | 2.375285 | 0.012826 |
| 3 | 23.526302% | 0.026163% | 2.368260 | 0.012067 |

Individual records:
- [seed 42](results/baseline/organsmnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/baseline/organsmnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/baseline/organsmnist/baseline_r3_e1_lr0p001/seed-44.json)

## Earlier Expanded Parameter Sweep (Seeds 42, 43, 44)

This earlier three-seed sweep is retained for historical comparison. The five-seed Validation-v2 sweep above is the clean reference for poisoning experiments.

The expanded parameter sweep varies the number of rounds, local epochs, and learning rate; see its table for each run's configuration. All runs below completed with 20 clients and passed the recorded run-validity check. Accuracy and standard deviation are across the three seeds at the final round of each run. Standard deviations are in percentage points.

| Dataset | Rounds | Local epochs | Learning rate | Mean final accuracy | Std. dev. |
|---------|-------:|-------------:|--------------:|--------------------:|----------:|
| OrganAMNIST | 3 | 1 | 0.001 | 18.52% | 0.06 pp |
| OrganAMNIST | 3 | 1 | 0.01 | 40.51% | 1.17 pp |
| OrganAMNIST | 3 | 1 | 0.1 | 43.89% | 5.28 pp |
| OrganAMNIST | 3 | 3 | 0.001 | 28.51% | 8.69 pp |
| OrganAMNIST | 5 | 1 | 0.001 | 20.62% | 3.70 pp |
| OrganAMNIST | 5 | 1 | 0.1 | **62.34%** | 1.59 pp |
| OrganAMNIST | 5 | 3 | 0.1 | 47.19% | 12.80 pp |
| OrganCMNIST | 3 | 1 | 0.001 | 19.86% | 4.29 pp |
| OrganCMNIST | 3 | 1 | 0.01 | 22.33% | 0.00 pp |
| OrganCMNIST | 3 | 1 | 0.1 | 44.33% | 8.77 pp |
| OrganCMNIST | 3 | 3 | 0.001 | 22.33% | 0.00 pp |
| OrganCMNIST | 5 | 1 | 0.001 | 22.33% | 0.00 pp |
| OrganCMNIST | 5 | 1 | 0.1 | 59.34% | 3.82 pp |
| OrganCMNIST | 5 | 3 | 0.1 | **73.48%** | 2.27 pp |
| OrganSMNIST | 3 | 1 | 0.001 | 23.53% | 0.03 pp |
| OrganSMNIST | 3 | 1 | 0.01 | 23.54% | 0.00 pp |
| OrganSMNIST | 3 | 1 | 0.1 | 32.70% | 5.82 pp |
| OrganSMNIST | 3 | 3 | 0.001 | 23.54% | 0.00 pp |
| OrganSMNIST | 5 | 1 | 0.001 | 23.54% | 0.00 pp |
| OrganSMNIST | 5 | 1 | 0.1 | 46.23% | 0.84 pp |
| OrganSMNIST | 5 | 3 | 0.1 | **55.96%** | 3.41 pp |

In this earlier sweep, the best observed setting differed by dataset: OrganAMNIST performed best with 5 rounds, 1 local epoch, and learning rate 0.1; OrganCMNIST and OrganSMNIST performed best with 5 rounds, 3 local epochs, and learning rate 0.1. These provisional three-seed results are superseded by Validation-v2 above and are not proof of global optima.

Increasing local epochs from 1 to 3 at 5 rounds and learning rate 0.1 improved mean accuracy on OrganCMNIST (59.34% to 73.48%) and OrganSMNIST (46.23% to 55.96%), but reduced OrganAMNIST mean accuracy (62.34% to 47.19%) and raised its across-seed standard deviation to 12.80 percentage points. Keep dataset-specific settings and preserve this variability in later comparisons.

## Observations from Earlier Runs
- The original 3-round, 1-epoch, 0.001 baseline reached a mean final accuracy of 18.52% on OrganAMNIST, 19.86% on OrganCMNIST, and 23.53% on OrganSMNIST.
- All recorded runs in the earlier parameter sweep completed with 20 client results per round and no recorded client failures.
- Evaluation loss decreased across rounds in the original OrganAMNIST baseline.
- Accuracy is not the only useful metric. Track per-class metrics and a confusion matrix in subsequent experiments, especially when comparing different client distributions or poisoned data.

## Next Steps
- Rerun the sweep with the corrected per-image loss and expanded per-round outputs under a new `protocol_<timestamp>` ID.
- Reconfirm the dataset settings from that protocol before using them as clean controls for poisoning experiments.
- Define one poisoning method and its parameters precisely, then change one poisoning factor at a time while holding the clean training setup, seeds, and client partitions fixed.
- Compare poisoned runs to their matching clean runs by seed; record accuracy and loss, and add per-class metrics/confusion matrices or attack success rate when relevant.
- After attack and defense settings are fixed, evaluate the selected final conditions on the test set without using test results for further tuning.
- For any additional dataset, establish a separate clean baseline; do not assume the Organ dataset settings transfer unchanged.
- Do not interpret the selected settings as universally optimal; they are the best observed validation configurations in this finite grid.
