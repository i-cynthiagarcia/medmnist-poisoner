
# Flower Federated Learning Baseline

## Experiment Setup
- Datasets: OrganAMNIST, OrganCMNIST, and OrganSMNIST (MedMNIST)
- Model: Convolutional Neural Network (CNN)
- Framework: Flower + PyTorch
- Aggregation: FedAvg

## Historical Baseline 

The settings below describe the originally reported OrganAMNIST historical
baseline. 

- Clients: 20
- Federated rounds: 3
- Local epochs: 1
- Learning rate: 0.001
- Batch size: 32

### Historical Results

These are the originally reported results. They are retained separately from
the summaries of runs recorded with references below.

The values below came from an earlier run and have not yet been verified against the current seeded implementation.

| Round | Accuracy | Evaluation Loss |
|-------|----------|-----------------|
| Initial | 10.60% | 2.3914 |
| 1 | 12.85% | 2.3758 |
| 2 | 18.48% | 2.3601 |
| 3 | 18.48% | 2.3400 |

## Results

### OrganAMNIST Seed 42-44 Summary
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 10.293621% | 0.725681% | 2.396808 | 0.003825 |
| 1 | 14.112949% | 3.812424% | 2.381487 | 0.001020 |
| 2 | 18.474144% | 0.006495% | 2.367219 | 0.003002 |
| 3 | 18.515394% | 0.064951% | 2.352745 | 0.007457 |

Individual records:
- [seed 42](results/organamnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/organamnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/organamnist/baseline_r3_e1_lr0p001/seed-44.json)

### OrganCMNIST Seed 42-44 Summary
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 10.365953% | 3.115142% | 2.393640 | 0.007745 |
| 1 | 13.222168% | 7.905748% | 2.388110 | 0.007652 |
| 2 | 16.869523% | 6.838005% | 2.382765 | 0.007564 |
| 3 | 19.855566% | 4.293586% | 2.377588 | 0.007500 |

Individual records:
- [seed 42](results/organcmnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/organcmnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/organcmnist/baseline_r3_e1_lr0p001/seed-44.json)

### OrganSMNIST Seed 42-44 Summary
| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 9.848571% | 2.792570% | 2.390166 | 0.014532 |
| 1 | 14.905026% | 7.964018% | 2.382573 | 0.013649 |
| 2 | 18.360334% | 7.333288% | 2.375285 | 0.012826 |
| 3 | 23.526302% | 0.026163% | 2.368260 | 0.012067 |

Individual records:
- [seed 42](results/organsmnist/baseline_r3_e1_lr0p001/seed-42.json)
- [seed 43](results/organsmnist/baseline_r3_e1_lr0p001/seed-43.json)
- [seed 44](results/organsmnist/baseline_r3_e1_lr0p001/seed-44.json)

## Expanded Parameter Sweep (Seeds 42, 43, 44)
The expanded parameter sweep varies the number of rounds, local epochs, and learning rate; see its table for each run's configuration. All runs below completed with 20 clients and passed the recorded run-validity
check. Accuracy and standard deviation are across the three seeds at the final
round of each run. Standard deviations are in percentage points.

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

The best observed setting differs by dataset: OrganAMNIST performed best with
5 rounds, 1 local epoch, and learning rate 0.1; OrganCMNIST and OrganSMNIST
performed best with 5 rounds, 3 local epochs, and learning rate 0.1. These are
provisional settings based on three seeds, not proof of global optima.

Increasing local epochs from 1 to 3 at 5 rounds and learning rate 0.1 improved
mean accuracy on OrganCMNIST (59.34% to 73.48%) and OrganSMNIST (46.23% to
55.96%), but reduced OrganAMNIST mean accuracy (62.34% to 47.19%) and raised
its across-seed standard deviation to 12.80 percentage points. Keep
dataset-specific settings and preserve this variability in later comparisons.

## Observations
- The original 3-round, 1-epoch, 0.001 baseline reached a mean final accuracy
  of 18.52% on OrganAMNIST, 19.86% on OrganCMNIST, and 23.53% on OrganSMNIST.
- All recorded parameter-sweep runs completed with 20 client results per
  round and no recorded client failures.
- Evaluation loss decreased across rounds in the original OrganAMNIST baseline.
- Best observed settings in this sweep: OrganAMNIST at 5 rounds, 1 local epoch,
  learning rate 0.1 (62.34%); OrganCMNIST at 5 rounds, 3 local epochs, learning
  rate 0.1 (73.48%); OrganSMNIST at 5 rounds, 3 local epochs, learning rate 0.1
  (55.96%).
- Accuracy is not the only useful metric. Track per-class metrics and a
  confusion matrix in subsequent experiments, especially when comparing
  different client distributions or poisoned data.

## Next Steps
- Establish clean baselines using the realistic client data distribution,
  starting with the best observed dataset-specific settings above. Keep the
  seeds, client count, model, and evaluation protocol fixed for comparability.
- Record accuracy, evaluation loss, per-class metrics, and confusion matrices
  for each clean run.
- Run poisoned-data experiments against those clean controls, changing the
  poisoning condition while holding the training configuration and seeds fixed.
- Establish a clean baseline separately for each new dataset; do not assume
  the Organ dataset settings transfer unchanged.
- Defer more combinations in the current parameter grid unless a later result
  indicates a specific tuning question. The current sweep is sufficient to
  proceed to realistic-distribution and poisoning experiments, but not to claim
  that the selected settings are universally optimal.
