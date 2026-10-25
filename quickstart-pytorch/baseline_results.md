
# Flower Federated Learning Baseline

## Dataset
- Dataset: OrganAMNIST (MedMNIST)
- Model: Convolutional Neural Network (CNN)
- Framework: Flower + PyTorch
- Aggregation: FedAvg

## Configuration
- Clients: 20
- Federated rounds: 3
- Local epochs: 1
- Learning rate: 0.001
- Batch size: 32

## Results

### Historical Quickstart Output
The values below came from an earlier run and have not yet been verified against the current seeded implementation.

| Round | Accuracy | Evaluation Loss |
|-------|----------|-----------------|
| Initial | 10.60% | 2.3914 |
| 1 | 12.85% | 2.3758 |
| 2 | 18.48% | 2.3601 |
| 3 | 18.48% | 2.3400 |

### Seed 42 Successful Rerun

All 20 client training results were received in each federated round.

| Round | Accuracy | Evaluation Loss | Client Train Loss |
|-------|----------|-----------------|-------------------|
| 0 | 9.674879% | 2.400985 | - |
| 1 | 18.477894% | 2.382435 | 2.393400 |
| 2 | 18.477894% | 2.364933 | 2.373386 |
| 3 | 18.590393% | 2.345442 | 2.352583 |

The structured record is stored in [results/organamnist/baseline_r3_e1_lr0p001/seed-42.json](results/organamnist/baseline_r3_e1_lr0p001/seed-42.json). 

### Seed 43 Successful Run

All 20 client training results were received in each federated round.

| Round | Accuracy | Evaluation Loss | Client Train Loss |
|-------|----------|-----------------|-------------------|
| 0 | 10.113624% | 2.393474 | - |
| 1 | 11.435482% | 2.381619 | 2.388559 |
| 2 | 18.477894% | 2.370620 | 2.376518 |
| 3 | 18.477894% | 2.360347 | 2.365225 |

The structured record is stored in [results/organamnist/baseline_r3_e1_lr0p001/seed-43.json](results/organamnist/baseline_r3_e1_lr0p001/seed-43.json).

### Seed 44 Successful Rerun

All 20 client training results were received in each federated round.

| Round | Accuracy | Evaluation Loss | Client Train Loss |
|-------|----------|-----------------|-------------------|
| 0 | 11.092361% | 2.395966 | - |
| 1 | 12.425470% | 2.380407 | 2.387347 |
| 2 | 18.466644% | 2.366105 | 2.371447 |
| 3 | 18.477894% | 2.352445 | 2.356523 |

The structured record is stored in [results/organamnist/baseline_r3_e1_lr0p001/seed-44.json](results/organamnist/baseline_r3_e1_lr0p001/seed-44.json). 

### Three-Seed Summary (Seeds 42, 43, 44)

| Round | Accuracy Mean | Accuracy Std Dev | Loss Mean | Loss Std Dev |
|-------|---------------|------------------|-----------|--------------|
| 0 | 10.293621% | 0.725681% | 2.396808 | 0.003825 |
| 1 | 14.112949% | 3.812424% | 2.381487 | 0.001020 |
| 2 | 18.474144% | 0.006495% | 2.367219 | 0.003002 |
| 3 | 18.515394% | 0.064951% | 2.352745 | 0.007457 |

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

## Observations
- All 20 clients successfully participated.
- No NaN values occurred during training.
- Evaluation loss decreased across rounds.
- Final global model accuracy: 18.48%.
- Total simulation time: 40.24 seconds.

## Next Steps
- Improve baseline accuracy.
- Evaluate additional training rounds.
- Implement data poisoning attacks.
- Compare clean and poisoned model performance.
