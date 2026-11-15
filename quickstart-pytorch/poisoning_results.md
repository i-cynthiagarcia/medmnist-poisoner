# Label-Flipping Results

This document tracks label-flipping protocol decisions, verification, and
high-level findings. The per-run JSON records are the source of truth for full
metrics, partition manifests, and artifact paths. Clean federated controls are
documented separately in [baseline_results.md](baseline_results.md).

## OrganAMNIST: kidney-left to kidney-right

**Status:** Completed exploratory trial; one seed only. Validation was used
throughout training, and test evaluation was disabled.

| Setting | Value |
|---------|-------|
| Dataset | OrganAMNIST, 28×28 |
| Source → target | kidney-left (ID 4) → kidney-right (ID 5) |
| Clients | 20 |
| Poisoned clients | 1 (5%); seed 42 selected client 2 |
| Flip fraction | 100% of client 2's eligible kidney-left training labels |
| Timing | Every training round, starting at round 1 |
| Training | 5 rounds, 1 local epoch, learning rate 0.1, batch size 32, FedAvg |
| Seed | 42 |

The poisoned client had 199 kidney-left examples, and all 199 labels were
flipped. That is about 0.58% of the 34,561-example training split. Validation
labels remained clean.

### Round-5 validation comparison

The matched clean control has the same seed and client partition as the saved
seed-42 baseline. Its per-round validation metrics match that baseline exactly.

| Condition | Accuracy | Loss | kidney-left recall | kidney-left → kidney-right |
|-----------|---------:|-----:|-------------------:|--------------------------:|
| Clean baseline/control | 81.11% | 0.6764 | 87.32% | 13/568 (2.29%) |
| 1 client, 100% flip | 87.18% | 0.5150 | 84.33% | 42/568 (7.39%) |
| Change | +6.07 pp | -0.1614 | -2.99 pp | +5.11 pp |

The target-directed error rate was 0% through round 3, 0.70% (4/568) at round
4, and 7.39% (42/568) at round 5. The increase is consistent with the intended
targeted direction in this trial. Overall accuracy was also higher in the
poisoned run, so the aggregate metric does not indicate a general degradation.
With one seed, these results are exploratory and do not establish a robust
attack effect.

### Label-transformation verification

The run record reports client 2 and 199 flipped examples. A replay using the
same OrganAMNIST training split, seed-42 client partition, and deterministic
transform confirmed that all 199 kidney-left labels in that client's 1,728
examples became kidney-right, no kidney-left labels remained in that client,
and every non-source label was unchanged. The seven focused data tests also
passed with:

```bash
python -m unittest discover -s tests -p 'test_data.py'
```

The original run did not persist a per-example before/after label audit. The
verification above reproduces the recorded transform against the exact
deterministic partition; it is not a saved snapshot of the tensors used during
the original Flower execution.

### Run records

- Poisoned run:
  [seed 42 JSON](results/poisoning/generation-v3/protocol-v1/run-1/organamnist/label_flip_r5_e1_lr0p100_clients1_f1p000_s4_t5/seed-42_run-13042088134346131756.json)
- Matched clean control:
  [seed 42 JSON](results/poisoning/generation-v3/protocol-v1/run-1/organamnist/label_flip_r5_e1_lr0p100_clients0_f0p000_s4_t5/seed-42_run-12640462282327131058.json)
- Saved clean baseline:
  [seed 42 JSON](results/baseline/generation-v3/protocol-v1/run-4/organamnist/baseline_r5_e1_lr0p100/seed-42_run-13016821332549798429.json)

All three runs completed five rounds with 20 client results per round and zero
client failures. The poisoned run's record also includes its confusion matrices,
plots, checkpoint references, and partition manifest.

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
per-example before/after label audit, so the count check does not independently
replay the tensor transformation.

Run records:

- OrganCMNIST seed 42: [clean control](results/poisoning/generation-v3/protocol-v1/run-3/organcmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/seed-42_run-17716134690215662785.json) · [poisoned run](results/poisoning/generation-v3/protocol-v1/run-3/organcmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/seed-42_run-13627381287354949253.json)
- OrganCMNIST seeds 43–46: [clean controls](results/poisoning/generation-v3/protocol-v1/run-5/organcmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/) · [poisoned runs](results/poisoning/generation-v3/protocol-v1/run-5/organcmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/)
- OrganSMNIST seed 42: [clean control](results/poisoning/generation-v3/protocol-v1/run-4/organsmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/seed-42_run-15574750315785558871.json) · [poisoned run](results/poisoning/generation-v3/protocol-v1/run-4/organsmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/seed-42_run-16040722783643213093.json)
- OrganSMNIST seeds 43–46: [clean controls](results/poisoning/generation-v3/protocol-v1/run-6/organsmnist/label_flip_r5_e3_lr0p100_clients0_f0p000_s4_t5/) · [poisoned runs](results/poisoning/generation-v3/protocol-v1/run-6/organsmnist/label_flip_r5_e3_lr0p100_clients1_f1p000_s4_t5/)

## Refinement notes

- Keep each poisoned run paired with a clean control using the same seed,
  client count, partition, and training settings.
- The OrganAMNIST seed-43–46 records are present under poisoning `run-2`;
  validate those before treating the OrganAMNIST effect as replicated.
- Change one attack factor at a time. Track targeted source-to-target
  misclassification, source-class recall, overall validation metrics, and
  label-transformation counts.
- Keep test evaluation disabled during refinement. Evaluate test data only
  after the protocol is fixed, and report it separately from validation.
- For every future result, record the generation/protocol/execution IDs, run
  JSON path, clean-control pairing, selected client IDs, flipped-example
  counts, label-audit outcome, and test-evaluation status.
