
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

| Round | Accuracy | Evaluation Loss |
|-------|----------|-----------------|
| Initial | 10.60% | 2.3914 |
| 1 | 12.85% | 2.3758 |
| 2 | 18.48% | 2.3601 |
| 3 | 18.48% | 2.3400 |

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
