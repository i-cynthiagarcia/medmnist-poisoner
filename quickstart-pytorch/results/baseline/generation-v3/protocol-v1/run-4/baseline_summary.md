## Current Reference: Complete Clean Baseline (generation-v3 / protocol-v1)

**Status:** The `run-4` execution contains 105 valid federated runs, with 5 seeds (42, 43, 44, 45, 46) for every tested configuration and dataset. No failed or incomplete federated runs need to be redone.

**Use:** Use these federated results as the current clean reference for poisoning comparisons. Centralized full-train runs, when present, are reported separately as comparators, not as federated results.

### Protocol and run integrity

- Datasets: OrganAMNIST, OrganCMNIST, and OrganSMNIST.
- Seeds: 42, 43, 44, 45, 46.
- Configurations: the seven combinations shown in the tables below, for each dataset.
- Federated training: 20 clients with FedAvg; every client returned a result in every round. All runs completed, passed the recorded validity check, and had zero client failures.
- Centralized comparisons: found 105 records; 3 contained invalid metrics, so no aggregate table was generated.
- Partition manifests: all 105 verify disjoint partitions and assignment of every training example exactly once.
- Metrics: validation accuracy and per-image cross-entropy loss are recorded at initialization and each training round. Test evaluation was disabled; `test_final` is null throughout.
- Run-specific JSON records and model checkpoints are in `results/baseline/generation-v3/protocol-v1/run-4/` and `models/baseline/generation-v3/protocol-v1/run-4/`.

### Federated sweep results

Accuracy is the mean final-round validation accuracy across 5 seeds; “±” is the population standard deviation in percentage points. Mean final loss is the per-image-weighted validation cross-entropy. The best observed configuration by mean validation accuracy for each dataset is bolded.

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
| OrganSMNIST | 5 | 1 | 0.1 | 46.13% ± 4.62 pp | 1.358 |
| **OrganSMNIST** | **5** | **3** | **0.1** | **70.48% ± 3.07 pp** | **0.882** |

### Centralized comparison data requiring attention

The centralized controls are not summarized because their records are incomplete or contain invalid metrics. Resolve these issues before using centralized comparisons:

- organamnist/centralized_r3_e1_lr0p100/seed-42: Non-finite final validation metrics for organamnist/centralized_r3_e1_lr0p100/seed-42.
- organamnist/centralized_r5_e1_lr0p100/seed-42: Non-finite final validation metrics for organamnist/centralized_r5_e1_lr0p100/seed-42.
- organamnist/centralized_r5_e3_lr0p100/seed-42: Non-finite final validation metrics for organamnist/centralized_r5_e3_lr0p100/seed-42.

### Candidate settings for poisoning comparisons

| Dataset | Rounds | Local epochs | Learning rate | Seeds |
|---------|-------:|-------------:|--------------:|-------|
| OrganAMNIST | 5 | 1 | 0.1 | 42, 43, 44, 45, 46 |
| OrganCMNIST | 5 | 3 | 0.1 | 42, 43, 44, 45, 46 |
| OrganSMNIST | 5 | 3 | 0.1 | 42, 43, 44, 45, 46 |

These are the best mean validation-accuracy settings among the seven tested configurations per dataset, not estimates of globally optimal settings. Keep each dataset's configuration fixed across its clean and poisoned runs; do not retune settings per attack condition. The shared validation set informed this selection, so reserve the test split for final evaluation after the protocol is locked.
