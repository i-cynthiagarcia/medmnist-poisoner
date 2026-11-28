# medmnist-poisoner

Evaluating the effects of data poisoning on a federated learning (FL) model for medical image classification, and testing robust aggregation defenses against it. (CSCI6330)

**Team:** Gabby Gutierrez, Norma Escalante, Cynthia Garcia, Stephanie Garcia

## Project Overview

Federated learning lets multiple clients (ex., hospitals) train a shared global model without pooling their raw data. That same decentralization makes FL vulnerbale to data poisoning: a malcious client can submit corrupted local updates to degrade or backdoor the global model.

We are simulating this setup using the [MedMNIST](https://medmnist.com) dataset and the [Flower](https://flower.ai/) FL framework. MedMNIST and Flower are used to train a clean federated baseline across multiple simulated clients, introduce one or more malicious/poisoned clients, measure damage to the model, then evaluate aggregation methods (for example, coordinate-wise median or trimmed mean, as a defense).

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

**Completed Baselines:**
- ✅ **Clean federated baseline** (generation-v3, protocol-v1) — 105 runs across OrganAMNIST, OrganCMNIST, and OrganSMNIST with 5 seeds each; best configurations identified per dataset with seed extensions (42–51)
- ✅ **Label-flipping experiment executions** (generation-v3, protocols v1–v5) — 659 run records and 660 partition manifests across the three datasets:
  - Protocol-v1: Single-client 100%-flip trials (3 datasets × 5 seeds)
  - Protocol-v2: Dose-response sweep, 1 attacker (OrganAMNIST, 10 seeds)
  - Protocol-v3: Factorial matrix 1,2,4,8 attackers × 4 flip fractions (OrganAMNIST, 10 seeds; 3 known numerical instabilities at 4-attacker level)
  - Protocol-v4: Stability pilot (OrganCMNIST/OrganSMNIST, confirming same training settings handle the extreme attack conditions)
  - Protocol-v5: Full factorial matrices (OrganCMNIST 223 run records including partial attempts, OrganSMNIST 170 run records; 10 seeds each in the complete matrices)

**Current milestone:**
- Label-flipping runs are preserved for defense evaluation; OrganAMNIST protocol-v3 analysis remains subject to three invalid cells
- 507 valid unique condition-seed cells across three datasets (OrganAMNIST: 167 from protocol-v3, OrganCMNIST: 170 from protocol-v5, OrganSMNIST: 170 from protocol-v5)
- Poisoning results and models are separated into `flipping` and reserved `overlay` directories. Overlay experiments and their protocol are forthcoming.
- Exploratory validation-only defense comparison completed for the four-client, 100%-flip OrganAMNIST attack. All ten median attack runs are valid, but only nine have completed median-clean controls; the comparison also reuses historical FedAvg records without source/environment manifests.
- The frozen 40-run OrganAMNIST comparison at learning rate 0.01 completed with no failed conditions. Both aggregations passed the 75% mean clean-accuracy floor; median had a 17.82 pp smaller mean attack-induced targeted-error increase.
- Expansion is paused for [class-level review](quickstart-pytorch/defense_results.md#class-level-validation-recall): some clean class recalls were weak despite the accuracy floor. Check OrganCMNIST and OrganSMNIST separately before adopting settings for them; reserve test evaluation for a separate decision.

**Documentation:**
- Detailed baseline results in [quickstart-pytorch/baseline_results.md](quickstart-pytorch/baseline_results.md)
- Comprehensive poisoning results and protocol progression in [quickstart-pytorch/poisoning_results.md](quickstart-pytorch/poisoning_results.md)
- Defense pilot and per-seed validation comparison in [quickstart-pytorch/defense_results.md](quickstart-pytorch/defense_results.md)
- Quickstart instructions and running experiments in [quickstart-pytorch/README.md](quickstart-pytorch/README.md)

## References

- J. Yang et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, no. 1, p. 41, 2023.
- Flower: A Friendly Federated Learning Framework. https://flower.ai/
- Tolpegin et al., "Data Poisoning Attacks Against Federated Learning Systems." https://arxiv.org/pdf/2007.08432
- Rodríguez-Barroso et al., "Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks." https://arxiv.org/pdf/2403.02983





