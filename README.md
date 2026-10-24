# medmnist-poisoner

Evaluating the effects of data poisoning on a federated learning (FL) model for medical image classification, and testing robust aggregation defenses against it. (CSCI6330)

**Team:** Gabby Gutierrez, Norma Escalante, Cynthia Garcia, Stephanie Garcia

## Project Overview

Federated learning lets multiple clients (ex., hospitals) train a shared global model without pooling their raw data. That same decentralization makes FL vulnerbale to data poisoning: a malcious client can submit corrupted local updates to degrade or backdoor the global model.

We are simulating this setup using the [MedMNIST](https://medmnist.com) dataset and the flower [Flower](https://flower.ai/) FL framework: is to train a clean federated baseline across multiple simulated clients, introduce one or more malicious/poisoned clients, meausre how much is the damage, then evaluate aggregation methods. For example, coordinate-wise median or trimmed mean, as a defense.

## Research Questions

Our revised focus now centers on measuring how poisoning affects the model rather than just if it does.
Research Questions:
1.) **How much** does a poisoned client degrade global model accuracy, and how does that scale with the degree/type of of poisoning. For example, like fraction of poisoned clients, or label-flip rate?
2.) Does introducing a **backdoor client** vs. just label-flipping or noise poisoning, produce a quality different failure mode?
3.) Can **robust aggregation** method reduce the measured degradation from (1) and (2)?

## Dataset

[MedMNIST v2](https://medmnist.com) — lightweight 2D biomedical image classification benchmarks. Candidate subsets under consideration:

- **OrganAMNIST / OrganCMNIST / OrganSMNIST** — abdominal CT, 11-class, used for the initial clean federated baseline
- **BreastMNIST** — breast ultrasound, binary-class, small (780 samples), for scaling experiments
- **TissueMNIST** — kidney cortex microscopy, 8-class, large (236k samples), for scaling experiments

## Approach

1. **Centralized baseline** — training a lightweight CNN on MedMNIST centrally as a reference point.
2. **Clean federated baseline** — partition the data across simulated clients (currently 20, uniform split) and train with Flower using FedAvg.
3. **Poisoned baseline** — introduce poisoned clients, like label-flipping, overlay/noise, or backdoor attacks, under both uniform and realistic client distributions. Also measure accuracy/degradation as a function of poisoning degree.
4. **Defense evaluation** — applying a robust aggregation, like coordinate-wise median or trimmed mean, and then compare recovered performance against the clean and poisoned baselines.

**Tools:** PyTorch, MedMNIST, Flower, NumPy, SciPy

## Status

**Milestone 1 (current):**
- Literature review on FL data-poisoning attacks and defenses completed
- Dataset and tool choices finalized (MedMNIST + Flower + PyTorch)
- Research question narrowed based on professor feedback
- Data preprocessing pipeline defined (MedMNIST train/test split, pixel normalization)
- 20-client uniform partitioning scheme defined
- Evaluation protocol for the clean federated baseline (FedAvg, no poisoning) defined

## References

- J. Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, no. 1, p. 41, 2023.
- Flower: A Friendly Federated Learning Framework. https://flower.ai/
- Tolpegin et al., "Data Poisoning Attacks Against Federated Learning Systems." https://arxiv.org/pdf/2007.08432
- Rodríguez-Barroso et al., "Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks." https://arxiv.org/pdf/2403.02983





