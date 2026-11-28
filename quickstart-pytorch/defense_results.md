# Defense results

## OrganCMNIST and OrganSMNIST: matched 40-run comparisons

Each validation-only comparison completed all 40 fresh conditions: seeds
42–51, FedAvg and coordinate-wise median, each clean and with four clients
flipping 100% of kidney-left labels to kidney-right. No condition failures
were reported. Every record reports five complete rounds, 20 client replies
per round, no client failures, finite reported validation metrics and a valid
run. All five round checkpoints and the final model are present for
every condition. Test evaluation was disabled.

All partition manifests verify disjoint, complete assignments. Within each
dataset and seed, the four conditions have matching client summaries; the
attacks use the seeded client selections and recorded full-flip counts.
Summaries contain image and class counts, not individual indices, so matching
assignments also relies on the seeded implementation and environment.

Both manifests record revision `41f1b20e31edc8f83a20a9b3c170c33cb653b7dd`
and matching source hashes, dependencies and environment settings. The
recorded source hashes and dataset archive checksums match the current files.
The local log reports no failed conditions, but contains Ray metrics-exporter
connection errors and `/tmp` object-store fallback warnings. These concern
telemetry and potential performance; the saved records meet the stated run
checks.

Results and manifests are in
[`protocol-v8/run-1/`](results/defense/generation-v1/protocol-v8/run-1/) for
OrganCMNIST and [`protocol-v8/run-2/`](results/defense/generation-v1/protocol-v8/run-2/)
for OrganSMNIST; their checkpoints and final models are under the matching
[`protocol-v8/` model directory](models/defense/generation-v1/protocol-v8/).
The local console log is `~/organc-organs-defense-comparison.8pe00C.log`.
The earlier pilot and diagnostic records are not pooled with these runs.

### Paired round-5 outcomes

Targeted error is the fraction of true kidney-left validation images predicted
as kidney-right. The primary metric is the per-seed difference in
attack-induced targeted error,
`(median attacked - median clean) - (FedAvg attacked - FedAvg clean)`;
negative favors median. Values are ten-seed means; rates and changes are
percentages or percentage points (pp), except for per-image loss.

| Dataset | Metric | FedAvg clean | FedAvg attacked | FedAvg change | Median clean | Median attacked | Median change |
|---|---|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | Accuracy (%) | 82.55 | 80.05 | -2.51 pp | 82.18 | 80.97 | -1.22 pp |
| OrganCMNIST | Targeted error (%) | 7.12 | 29.47 | +22.35 pp | 7.35 | 19.77 | +12.42 pp |
| OrganCMNIST | Kidney-left recall (%) | 58.41 | 7.95 | -50.45 pp | 58.26 | 26.82 | -31.44 pp |
| OrganCMNIST | Macro recall (%) | 76.37 | 72.14 | -4.23 pp | 75.84 | 73.80 | -2.04 pp |
| OrganCMNIST | Loss | 0.5761 | 0.6001 | +0.0240 | 0.5794 | 0.5906 | +0.0111 |
| OrganSMNIST | Accuracy (%) | 58.53 | 57.54 | -0.98 pp | 58.03 | 57.38 | -0.64 pp |
| OrganSMNIST | Targeted error (%) | 15.93 | 43.50 | +27.57 pp | 13.86 | 33.50 | +19.64 pp |
| OrganSMNIST | Kidney-left recall (%) | 30.64 | 0.21 | -30.43 pp | 30.93 | 6.50 | -24.43 pp |
| OrganSMNIST | Macro recall (%) | 50.86 | 49.23 | -1.63 pp | 50.37 | 49.23 | -1.15 pp |
| OrganSMNIST | Loss | 1.0803 | 1.0900 | +0.0098 | 1.0890 | 1.0902 | +0.0012 |

Both methods passed both frozen ten-seed clean floors on both datasets.
Median's mean clean accuracy was 0.37 pp lower than FedAvg on OrganCMNIST
and 0.50 pp lower on OrganSMNIST. The floors apply to means, not individual
seeds: for example, both methods fell below both OrganCMNIST floors on seed
45 and both OrganSMNIST floors on seed 47. Passing the means does not remove
those individual-seed limitations.

The mean targeted-error interaction was **-9.92 pp** for OrganCMNIST and
**-7.93 pp** for OrganSMNIST. It favored median on all ten OrganCMNIST seeds
and nine of ten OrganSMNIST seeds; OrganSMNIST seed 44 favored FedAvg by
1.43 pp. Median reduced the mean attack-induced targeted-error increase, but
attacked kidney-left recall remained low, especially on OrganSMNIST.

| Dataset | Seed | FedAvg clean accuracy | Median clean accuracy | FedAvg targeted-error change | Median targeted-error change | Interaction |
|---|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | 42 | 82.23% | 81.61% | +25.00 pp | +14.39 pp | -10.61 pp |
| OrganCMNIST | 43 | 85.08% | 84.82% | +27.27 pp | +16.67 pp | -10.61 pp |
| OrganCMNIST | 44 | 83.44% | 83.53% | +25.76 pp | +14.39 pp | -11.36 pp |
| OrganCMNIST | 45 | 79.35% | 79.43% | +18.18 pp | +12.12 pp | -6.06 pp |
| OrganCMNIST | 46 | 80.39% | 79.60% | +10.61 pp | +6.06 pp | -4.55 pp |
| OrganCMNIST | 47 | 81.98% | 81.90% | +21.21 pp | +12.12 pp | -9.09 pp |
| OrganCMNIST | 48 | 83.53% | 82.98% | +12.88 pp | +9.09 pp | -3.79 pp |
| OrganCMNIST | 49 | 85.08% | 84.66% | +27.27 pp | +13.64 pp | -13.64 pp |
| OrganCMNIST | 50 | 84.11% | 83.11% | +31.82 pp | +17.42 pp | -14.39 pp |
| OrganCMNIST | 51 | 80.35% | 80.18% | +23.48 pp | +8.33 pp | -15.15 pp |
| OrganSMNIST | 42 | 62.28% | 61.70% | +32.14 pp | +17.14 pp | -15.00 pp |
| OrganSMNIST | 43 | 62.77% | 62.40% | +37.14 pp | +25.71 pp | -11.43 pp |
| OrganSMNIST | 44 | 62.77% | 63.05% | +37.86 pp | +39.29 pp | +1.43 pp |
| OrganSMNIST | 45 | 55.26% | 54.28% | +20.71 pp | +17.14 pp | -3.57 pp |
| OrganSMNIST | 46 | 52.08% | 52.12% | +25.71 pp | +12.86 pp | -12.86 pp |
| OrganSMNIST | 47 | 50.65% | 49.35% | +12.14 pp | +7.86 pp | -4.29 pp |
| OrganSMNIST | 48 | 57.06% | 56.44% | +27.86 pp | +16.43 pp | -11.43 pp |
| OrganSMNIST | 49 | 56.89% | 56.48% | +22.14 pp | +17.86 pp | -4.29 pp |
| OrganSMNIST | 50 | 64.64% | 63.54% | +35.71 pp | +22.14 pp | -13.57 pp |
| OrganSMNIST | 51 | 60.89% | 60.89% | +24.29 pp | +20.00 pp | -4.29 pp |

Values are rounded independently; means and changes use unrounded values.
These results describe the specified attack on the shared validation set,
not all attack settings or unseen test data.

### OrganC and OrganS class-level validation recall

Each entry is the mean recall percentage across ten seeds. Support is the
number of validation examples for that class.

| OrganCMNIST class | Support | FedAvg clean | FedAvg attacked | Median clean | Median attacked |
|---|---:|---:|---:|---:|---:|
| Bladder | 191 | 83.04 | 83.35 | 82.57 | 82.93 |
| Femur-left | 102 | 69.31 | 69.80 | 67.06 | 67.35 |
| Femur-right | 96 | 40.94 | 39.38 | 40.00 | 42.71 |
| Heart | 202 | 86.19 | 86.68 | 85.79 | 86.53 |
| Kidney-left | 132 | 58.41 | 7.95 | 58.26 | 26.82 |
| Kidney-right | 157 | 71.59 | 75.54 | 71.91 | 74.90 |
| Liver | 429 | 94.43 | 94.31 | 94.36 | 94.22 |
| Lung-left | 347 | 93.03 | 92.65 | 92.85 | 92.39 |
| Lung-right | 352 | 92.02 | 92.07 | 92.07 | 91.99 |
| Pancreas | 179 | 73.58 | 73.80 | 73.46 | 73.97 |
| Spleen | 205 | 77.56 | 78.05 | 75.90 | 77.95 |

| OrganSMNIST class | Support | FedAvg clean | FedAvg attacked | Median clean | Median attacked |
|---|---:|---:|---:|---:|---:|
| Bladder | 188 | 52.66 | 54.20 | 50.00 | 51.70 |
| Femur-left | 104 | 41.92 | 44.33 | 40.67 | 40.77 |
| Femur-right | 95 | 25.05 | 21.16 | 28.00 | 27.58 |
| Heart | 246 | 58.29 | 58.21 | 56.71 | 57.20 |
| Kidney-left | 140 | 30.64 | 0.21 | 30.93 | 6.50 |
| Kidney-right | 159 | 46.10 | 61.38 | 44.72 | 55.47 |
| Liver | 491 | 88.45 | 88.35 | 88.86 | 88.57 |
| Lung-left | 261 | 35.29 | 33.68 | 35.10 | 34.98 |
| Lung-right | 275 | 84.95 | 85.56 | 84.55 | 84.84 |
| Pancreas | 280 | 66.14 | 64.96 | 65.86 | 65.29 |
| Spleen | 213 | 30.00 | 29.53 | 28.73 | 28.59 |

OrganSMNIST clean femur-right recall was zero on seeds 42, 45, 47, 49 and 51
under both aggregations, and varied up to 95.79% for FedAvg and 94.74% for
median. Both femur classes had zero clean recall on seeds 45 and 47 for both
methods. The ten-seed femur-right means are higher than in the three-seed
pilot, but conceal severe seed dependence. OrganCMNIST also had weak
femur-right recall on some seeds: FedAvg was zero on seed 51, and median was
2.08% on seed 47.

| OrganSMNIST clean femur-right recall (%) | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.00 | 95.79 | 2.11 | 0.00 | 56.84 | 0.00 | 94.74 | 0.00 | 1.05 | 0.00 |
| Median | 0.00 | 94.74 | 2.11 | 0.00 | 87.37 | 0.00 | 94.74 | 0.00 | 1.05 | 0.00 |

For OrganSMNIST, attacked kidney-left recall was zero on eight seeds with
FedAvg and four with median. Femur-right is not the attack target; its poor
clean recall is a separate model-quality limitation. Changes in other classes
under attack should not alone be interpreted as an overall benefit. The
OrganSMNIST comparison remains exploratory; the frozen floors and validation
results are not guarantees of generalization or clinical suitability.

## OrganCMNIST and OrganSMNIST defense pilots

The initial validation-only pilots each completed 12 conditions: seeds 42, 49
and 50, with FedAvg and coordinate-wise median, each clean and with four
clients flipping 100% of kidney-left labels to kidney-right. Both used 20
clients, five rounds, one local epoch, batch size 32 and learning rate 0.01.
All 24 conditions are valid; no condition failures were reported.

The subsequent three-epoch clean diagnostic and matched attack pilot each
completed six conditions per dataset: the same seeds and aggregations, with
all other listed settings unchanged. All 24 three-epoch conditions are valid.
Across the pilots and diagnostic, every record reports five complete rounds,
20 client replies per round, zero client failures, finite metrics and model
parameters, and a final model and five round checkpoints. Test evaluation was
disabled. The logs contain Ray metrics-exporter connection errors and `/tmp`
object-store fallback warnings; the saved training results nonetheless meet
the stated checks.

The one-epoch pilots' recorded partition summaries agree across the matched
conditions, and the attacked runs select the same seeded clients and report
the same flipped-example counts for each aggregation. The three-epoch
diagnostic's summaries agree across aggregations, with the corresponding
one-epoch controls, and with the three-epoch attacks. The three-epoch attacks
also use the expected seeded attackers and flip all recorded kidney-left
examples on those clients. These summaries record counts, not individual
indices; matching assignments also relies on the seeded implementation and
environment.

Both pilots and the diagnostic recorded revision
`d90aed8de1e55527a3fc6593b7b20bc8d3e151f5`, the same source hashes, Python,
dependencies, platform, CUDA and thread settings, and matching per-dataset
archive checksums. The diagnostic captured its environment at startup; unlike
the pilot runner, its command did not recheck for environment drift during
execution. Current source hashes and archive checksums match the manifests,
but this does not independently prove nothing changed during the diagnostic.
The attack pilot recorded revision `e0859e3f4df09b901e13344433f776a59ec7e54b`.
Its source hashes, archive checksums, and captured environment match the
diagnostic, apart from the editable project's recorded revision. The attack
command also captured its environment only at startup; neither command
monitored for mid-run changes.

The initial results and manifests are in
[`protocol-v5/run-1/`](results/defense/generation-v1/protocol-v5/run-1/) for
OrganCMNIST and [`protocol-v5/run-2/`](results/defense/generation-v1/protocol-v5/run-2/)
for OrganSMNIST; models are under the matching
[`protocol-v5/` model directory](models/defense/generation-v1/protocol-v5/).
The diagnostic's records and manifest are in
[`protocol-v6/run-1/`](results/defense/generation-v1/protocol-v6/run-1/),
with [models and checkpoints](models/defense/generation-v1/protocol-v6/run-1/).
The attack records and manifest are in
[`protocol-v7/run-1/`](results/defense/generation-v1/protocol-v7/run-1/),
with [models and checkpoints](models/defense/generation-v1/protocol-v7/run-1/).
The local logs are `~/organc-organs-defense-pilot.Vbt9cR.log`,
`~/organc-organs-clean-diagnostic.WNYiGD.log`, and
`~/organc-organs-attack-pilot.LuXpPN.log`.

### One-epoch matched pilot

Values are means across three seeds at round 5. Targeted error is the fraction
of true kidney-left validation images predicted as kidney-right; loss is
per-image cross-entropy.

| Dataset | Aggregation | Clean accuracy | Attacked accuracy | Clean loss | Attacked loss | Clean / attacked targeted error | Clean / attacked kidney-left recall |
|---|---|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | FedAvg | 34.27% | 34.55% | 1.9118 | 1.9001 | 0% / 0% | 0% / 0% |
| OrganCMNIST | Median | 34.28% | 34.82% | 1.9124 | 1.9017 | 0% / 0% | 0% / 0% |
| OrganSMNIST | FedAvg | 23.36% | 23.37% | 2.1251 | 2.1207 | 0% / 0% | 0% / 0% |
| OrganSMNIST | Median | 23.36% | 23.36% | 2.1298 | 2.1250 | 0% / 0% | 0% / 0% |

| Dataset | Seed | FedAvg clean accuracy | FedAvg attacked accuracy | Median clean accuracy | Median attacked accuracy |
|---|---:|---:|---:|---:|---:|
| OrganCMNIST | 42 | 44.11% | 44.15% | 43.98% | 43.85% |
| OrganCMNIST | 49 | 17.93% | 17.98% | 17.93% | 17.93% |
| OrganCMNIST | 50 | 40.76% | 41.51% | 40.93% | 42.68% |
| OrganSMNIST | 42 | 30.02% | 30.06% | 30.02% | 30.02% |
| OrganSMNIST | 49 | 20.02% | 20.02% | 20.02% | 20.02% |
| OrganSMNIST | 50 | 20.02% | 20.02% | 20.02% | 20.02% |

Neither kidney class was predicted in any clean or attacked condition on
either dataset. The per-seed attack-induced targeted-error change and its
median-minus-FedAvg difference were therefore zero throughout. This is
not evidence of defense success: the models did not recognize the source
class even without attack. Mean clean macro recall was 19.24% / 19.25%
for OrganCMNIST and 11.80% / 11.79% for OrganSMNIST (FedAvg / median).

### Three-epoch clean diagnostic

The initial diagnostic contained clean controls only; its subsequently
completed matched attacks are analyzed below. The three-seed mean final
accuracy and macro recall, with classes weighted equally for macro recall,
were:

| Dataset | FedAvg accuracy | Median accuracy | FedAvg macro recall | Median macro recall | FedAvg loss | Median loss |
|---|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | 83.81% | 83.12% | 78.71% | 77.87% | 0.5463 | 0.5516 |
| OrganSMNIST | 61.27% | 60.58% | 54.86% | 54.08% | 1.0213 | 1.0308 |

Median's mean clean accuracy was 0.68 percentage points lower than FedAvg's
on OrganCMNIST and 0.69 points lower on OrganSMNIST. Differences and means
use unrounded values. Compared with the one-epoch pilots, both aggregations
had substantially higher clean accuracy and macro recall, and nonzero
kidney-left recall. The settings are separate protocols and must not be pooled.

| Dataset | Seed | FedAvg accuracy | Median accuracy | FedAvg kidney-left recall | Median kidney-left recall | FedAvg targeted error | Median targeted error |
|---|---:|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | 42 | 82.23% | 81.61% | 65.15% | 63.64% | 3.79% | 4.55% |
| OrganCMNIST | 49 | 85.08% | 84.66% | 62.88% | 61.36% | 9.09% | 9.09% |
| OrganCMNIST | 50 | 84.11% | 83.11% | 59.85% | 56.06% | 11.36% | 14.39% |
| OrganSMNIST | 42 | 62.28% | 61.70% | 52.14% | 52.14% | 10.71% | 10.71% |
| OrganSMNIST | 49 | 56.89% | 56.48% | 23.57% | 22.14% | 14.29% | 11.43% |
| OrganSMNIST | 50 | 64.64% | 63.54% | 42.86% | 42.86% | 15.00% | 14.29% |

#### Clean class-level validation recall

Each recall is the mean percentage across three seeds; support is the number
of validation examples for that class.

| Class | OrganC support | OrganC FedAvg | OrganC median | OrganS support | OrganS FedAvg | OrganS median |
|---|---:|---:|---:|---:|---:|---:|
| Bladder | 191 | 81.33 | 80.28 | 188 | 46.63 | 43.44 |
| Femur-left | 102 | 83.66 | 77.78 | 104 | 92.31 | 93.59 |
| Femur-right | 96 | 46.53 | 50.00 | 95 | 0.35 | 0.35 |
| Heart | 202 | 86.63 | 86.47 | 246 | 67.21 | 64.91 |
| Kidney-left | 132 | 62.63 | 60.35 | 140 | 39.52 | 39.05 |
| Kidney-right | 157 | 73.04 | 73.67 | 159 | 49.90 | 47.80 |
| Liver | 429 | 94.56 | 94.41 | 491 | 89.14 | 89.27 |
| Lung-left | 347 | 94.14 | 93.47 | 261 | 36.78 | 33.33 |
| Lung-right | 352 | 91.76 | 91.76 | 275 | 86.06 | 87.76 |
| Pancreas | 179 | 73.18 | 72.44 | 280 | 66.43 | 68.45 |
| Spleen | 205 | 78.37 | 75.93 | 213 | 29.11 | 26.92 |

OrganSMNIST femur-right recall was zero for seeds 42 and 49 and 1.05% for
seed 50 under both aggregations. Its kidney-left recall varied from 22.14%
to 52.14% across the six clean runs. OrganCMNIST femur-right recall also
varied: for example, median achieved 22.92% on seed 42 and 91.67% on seed
49. These weaknesses are not captured by overall accuracy.

### Decision and next steps

The three-epoch candidate was more informative than the one-epoch setting,
but three seeds alone did not establish stability. The subsequently completed
40-condition comparisons are summarized above. OrganSMNIST remains exploratory
with a known femur-right limitation. Broader matrices and test evaluation
remain separate decisions.

### Three-epoch matched attack pilot

The four-client, 100%-flip attack maps kidney-left (class 4) to kidney-right
(class 5); femur-right is **not** an attack source or target. It is included
in the class-level review to expose weak clean or collateral performance.
This is the same targeted kidney attack evaluated in the one-epoch pilot.

The table gives three-seed means at round 5. The primary metric is the
per-seed difference in attack-induced targeted error,
`(median attacked - median clean) - (FedAvg attacked - FedAvg clean)`;
negative favors median. Changes and means use unrounded values, and rates
and differences are percentages or percentage points (pp), except for loss.

| Dataset | Metric | FedAvg clean | FedAvg attacked | FedAvg change | Median clean | Median attacked | Median change |
|---|---|---:|---:|---:|---:|---:|---:|
| OrganCMNIST | Accuracy (%) | 83.81 | 81.10 | -2.70 pp | 83.12 | 82.07 | -1.06 pp |
| OrganCMNIST | Targeted error (%) | 8.08 | 36.11 | +28.03 pp | 9.34 | 24.49 | +15.15 pp |
| OrganCMNIST | Kidney-left recall (%) | 62.63 | 7.58 | -55.05 pp | 60.35 | 26.01 | -34.34 pp |
| OrganCMNIST | Loss | 0.5463 | 0.5694 | +0.0231 | 0.5516 | 0.5596 | +0.0080 |
| OrganSMNIST | Accuracy (%) | 61.27 | 59.65 | -1.62 pp | 60.58 | 59.95 | -0.63 pp |
| OrganSMNIST | Targeted error (%) | 13.33 | 43.33 | +30.00 pp | 12.14 | 31.19 | +19.05 pp |
| OrganSMNIST | Kidney-left recall (%) | 39.52 | 0.71 | -38.81 pp | 39.05 | 13.10 | -25.95 pp |
| OrganSMNIST | Loss | 1.0213 | 1.0359 | +0.0146 | 1.0308 | 1.0363 | +0.0055 |

| Dataset | Seed | FedAvg targeted-error change | Median targeted-error change | Median minus FedAvg change |
|---|---:|---:|---:|---:|
| OrganCMNIST | 42 | +25.00 pp | +14.39 pp | -10.61 pp |
| OrganCMNIST | 49 | +27.27 pp | +13.64 pp | -13.64 pp |
| OrganCMNIST | 50 | +31.82 pp | +17.42 pp | -14.39 pp |
| OrganSMNIST | 42 | +32.14 pp | +17.14 pp | -15.00 pp |
| OrganSMNIST | 49 | +22.14 pp | +17.86 pp | -4.29 pp |
| OrganSMNIST | 50 | +35.71 pp | +22.14 pp | -13.57 pp |

The mean paired difference was -12.88 pp for OrganCMNIST and -10.95 pp for
OrganSMNIST, negative on all three seeds. Median reduced the measured
attack-induced targeted-error increase but did not prevent substantial
kidney-left recall loss. These are exploratory validation results for one
attack setting and three selected seeds, not a general robustness claim.

#### Attacked class-level validation recall

The table gives mean recall percentages across three seeds. The corresponding
clean recalls and class supports are in the clean diagnostic table above.

| Class | OrganC FedAvg attacked | OrganC median attacked | OrganS FedAvg attacked | OrganS median attacked |
|---|---:|---:|---:|---:|
| Bladder | 80.98 | 79.23 | 48.94 | 45.04 |
| Femur-left | 83.33 | 82.68 | 91.99 | 93.27 |
| Femur-right | 45.14 | 47.57 | 0.70 | 0.70 |
| Heart | 86.96 | 87.13 | 66.67 | 65.31 |
| Kidney-left | 7.58 | 26.01 | 0.71 | 13.10 |
| Kidney-right | 78.77 | 78.13 | 64.78 | 58.70 |
| Liver | 94.02 | 94.17 | 89.27 | 89.07 |
| Lung-left | 93.66 | 93.18 | 33.08 | 35.89 |
| Lung-right | 91.86 | 91.86 | 87.64 | 87.39 |
| Pancreas | 73.00 | 73.00 | 65.12 | 66.19 |
| Spleen | 80.65 | 81.63 | 27.39 | 27.54 |

OrganSMNIST femur-right recall was already 0.35% on average under both
clean aggregations and was 0.70% when attacked; the class has 95 validation
examples. By seed, clean to attacked recalls were 0% to 0%, 0% to 0%, and
1.05% to 2.11% for FedAvg, and 0% to 1.05%, 0% to 0%, and 1.05% to 1.05%
for median (seeds 42, 49, 50). These small differences do not establish
an improvement. For OrganSMNIST seed 49, attacked kidney-left recall fell to
zero under both methods.

#### Frozen OrganCMNIST and OrganSMNIST comparisons

The completed comparisons above followed this frozen protocol:

For each dataset, run all 40 conditions **fresh**: seeds 42–51, FedAvg and
median, each clean and with four of 20 clients flipping 100% of kidney-left
labels to kidney-right. Use five rounds, three local epochs, learning rate
0.01 and batch size 32. Require all 20 clients and keep test evaluation
disabled. Do not reuse the preliminary clean or attack records as cells.

The frozen clean-performance floors are **80% accuracy and 50% kidney-left
recall for OrganCMNIST**, and **55% accuracy and 25% kidney-left recall for
OrganSMNIST**. Apply each floor to the unrounded mean of the ten clean seed
results separately for each aggregation, and only assess that aggregation
when all ten clean cells are valid. Report individual seeds and all class
recalls. No all-class floor or median-versus-FedAvg clean-deficit threshold is
frozen. In particular, report the OrganSMNIST femur-right weakness; its floor
does not establish general model or clinical suitability. These practical
floors were informed by the three-seed pilot, not independent test data.

The primary attack metric is the per-seed difference in attack-induced targeted
error, `(median attacked - median clean) - (FedAvg attacked - FedAvg clean)`;
negative favors median. Report its mean and each paired seed, plus each
method's attack change, clean accuracy difference, kidney-left recall,
validation loss, and class-level performance. No attack-effect threshold is
frozen.

Require all 40 valid conditions for a complete comparison on each dataset.
Continue after condition failures and report them; do not silently exclude
numerical failures or treat unpaired cells as pairs. Retry at most once only
for a documented infrastructure failure, with the same seed, configuration,
source and environment, in a separate execution preserving both attempts.
The runner does not retry automatically. If cells remain invalid, report
the incomplete comparison and any complete pairs as exploratory.

The [comparison runner and instructions](README.md#organcmnist-and-organsmnist-frozen-comparisons)
create separate new execution directories per dataset. A floor failure is a
reported scientific outcome, not a condition-execution failure.

## Matched 0.01 comparison: 40 runs

The OrganAMNIST validation-only comparison completed all 40 conditions for seeds 42–51:
FedAvg and coordinate-wise median, each clean and with four clients flipping
100% of their kidney-left labels to kidney-right. Every record reports five
complete rounds, 20 successful replies per round, no client failures, finite
metrics and parameters, and a valid run. All partition manifests report
disjoint, complete assignments. Within each seed, all four conditions used
the same partition, and the two attacks selected the same clients. All five
recorded round checkpoints and the final model are present for every run.
Test evaluation was disabled.

The runner log reports no failed conditions. It contains Ray metrics-exporter
connection errors, which affect telemetry; the experiment records nevertheless
show complete, valid runs. The execution manifest records revision
`489e4ad610b648f494fd9c9d6cfa398eddce221e` and a clean starting working tree.
Its training-source hashes, dataset checksum, Python, platform, CUDA and
thread settings match the 0.01 pilot; installed dependencies also match
except for the editable project's recorded revision.

Results and the manifest are in
[`protocol-v3/run-2/`](results/defense/generation-v1/protocol-v3/run-2/);
models and checkpoints are in the matching
[`models/` execution](models/defense/generation-v1/protocol-v3/run-2/).
The local console log is `~/organa-lr001-final.LN327F.log`.

### Paired round-5 outcomes

Targeted error is the fraction of true kidney-left validation images predicted
as kidney-right. The attack-effect difference is
`(median attacked - median clean) - (FedAvg attacked - FedAvg clean)`; a
negative value favors median. Rates and changes are in percentage points
(pp), except for the per-image cross-entropy loss.

| Metric | FedAvg clean | FedAvg attacked | FedAvg change | Median clean | Median attacked | Median change |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy (%) | 75.58 | 73.56 | -2.01 pp | 75.35 | 75.40 | +0.05 pp |
| Targeted error (%) | 6.14 | 31.69 | +25.55 pp | 6.36 | 14.08 | +7.73 pp |
| Kidney-left recall (%) | 75.86 | 24.96 | -50.90 pp | 75.46 | 61.13 | -14.33 pp |
| Loss | 0.7484 | 0.7720 | +0.0236 | 0.7474 | 0.7552 | +0.0078 |

Both aggregations passed the frozen 75% mean clean-accuracy floor. The
median-minus-FedAvg clean accuracy difference was -0.23 pp on average. The
mean targeted-error attack-effect difference was **-17.82 pp**, and was
negative for every seed:

| Seed | FedAvg clean accuracy | Median clean accuracy | FedAvg targeted-error change | Median targeted-error change | Attack-effect difference |
|---:|---:|---:|---:|---:|---:|
| 42 | 75.26% | 75.43% | +11.80 pp | +4.93 pp | -6.87 pp |
| 43 | 77.58% | 77.66% | +37.32 pp | +10.92 pp | -26.41 pp |
| 44 | 76.31% | 76.92% | +14.96 pp | +4.93 pp | -10.04 pp |
| 45 | 70.28% | 70.37% | +32.39 pp | +5.11 pp | -27.29 pp |
| 46 | 78.17% | 77.68% | +19.37 pp | +9.33 pp | -10.04 pp |
| 47 | 75.60% | 74.24% | +26.76 pp | +5.46 pp | -21.30 pp |
| 48 | 77.34% | 76.52% | +14.26 pp | +4.93 pp | -9.33 pp |
| 49 | 74.13% | 73.86% | +28.35 pp | +5.63 pp | -22.71 pp |
| 50 | 80.26% | 80.08% | +21.83 pp | +10.92 pp | -10.92 pp |
| 51 | 70.82% | 70.71% | +48.42 pp | +15.14 pp | -33.27 pp |

Values are rounded independently; changes and means use unrounded values.
These results apply to the specified four-client attack and shared validation
set, not to all attack settings or unseen test data.

### Class-level validation recall

Each entry is the mean recall percentage across ten seeds. Support is the
number of validation examples for that class.

| Class | Support | FedAvg clean | FedAvg attacked | Median clean | Median attacked |
|---|---:|---:|---:|---:|---:|
| Bladder | 321 | 30.31 | 35.17 | 30.53 | 33.24 |
| Femur-left | 233 | 67.17 | 65.45 | 65.11 | 62.06 |
| Femur-right | 225 | 46.80 | 45.02 | 40.31 | 45.64 |
| Heart | 392 | 93.47 | 95.41 | 93.78 | 94.95 |
| Kidney-left | 568 | 75.86 | 24.96 | 75.46 | 61.13 |
| Kidney-right | 637 | 50.77 | 58.57 | 50.68 | 53.44 |
| Liver | 1033 | 97.11 | 97.05 | 97.07 | 97.17 |
| Lung-left | 1033 | 90.67 | 90.78 | 90.78 | 90.43 |
| Lung-right | 1009 | 85.56 | 86.02 | 85.56 | 86.04 |
| Pancreas | 529 | 80.28 | 80.02 | 80.43 | 80.59 |
| Spleen | 511 | 38.77 | 56.32 | 39.55 | 48.57 |

Mean clean macro recall, giving each class equal weight, was 68.80% for
FedAvg and 68.11% for median. Both aggregations had zero clean recall for
both femur classes on seeds 45 and 51; seed 45 also had zero bladder recall.
This does not invalidate the frozen accuracy criterion, but the criterion
does not capture these class-level weaknesses. Changes in other classes under
attack should not alone be interpreted as an overall benefit.

The historical 0.1 FedAvg clean controls on these same seeds averaged 80.30%
accuracy and 71.66% macro recall, versus 75.58% and 68.80% at 0.01. Class
tradeoffs differ: for example, historical bladder mean recall was 85.05%
versus 30.31%, while femur-left was 24.29% versus 67.17%. The historical
controls are from a different protocol and are context, not a controlled
estimate of the effect of learning rate; do not pool them.

The clean class-level weaknesses remain a limitation of this setting; passing
the accuracy floor does not resolve them. A longer-training diagnostic remains
an option under a separate protocol. If a changed setting is adopted, rerun
the full matched comparison rather than altering these results.

### Planned OrganAMNIST attack matrix

The next sweep retains the frozen 0.01 comparison settings and seeds 42–51.
For each aggregation and seed, vary poisoned clients over 1, 2, 4 and 8 of
20, and the fraction of each poisoned client's kidney-left labels flipped to
kidney-right over 25%, 50%, 75% and 100%. The seeded client and example
selection makes lower counts and doses nested within higher ones for each
seed; realized flip counts are rounded per client.

The full matrix has 320 attacked conditions plus 20 matched clean controls.
The existing 20 clean controls and 20 four-client/100% attacks from
`protocol-v3/run-2` are reused, leaving 300 new attacked conditions. This
reuse requires the same training source, dataset and environment; results
from the historical 0.1 protocol are not interchangeable. The
[sweep runner and instructions](README.md#organamnist-defense-matrix-sweep)
check the reference and environment before starting, retain a manifest for
each new attack setting, and report failures without silently dropping cells.

For each matrix cell, report paired attack-induced targeted-error changes
and their median-minus-FedAvg difference, with class recall, accuracy, loss
and per-seed outcomes. Treat the matrix as descriptive unless a multiple-
comparison analysis is specified in advance. These results remain
validation-only and retain the known clean class-level limitations.
OrganCMNIST and OrganSMNIST completed their own preliminary checks and
40-condition comparisons, summarized above.

### Planned OrganCMNIST and OrganSMNIST attack matrices

The proposed expansions use the same attack matrix, aggregations and seeds as
OrganAMNIST, with each dataset's frozen three-epoch settings. For each
dataset, reuse the 20 matched clean controls and 20 four-client/100% attacks
from its completed `protocol-v8` comparison, leaving 300 new attacked
conditions. The new attacks use `generation-v1/protocol-v9`. Reuse is
conditional on reference records, artifacts, source hashes, dataset archive
and environment passing preflight checks.

Analyze each setting using matched within-seed attack changes and the same
median-minus-FedAvg interaction as the original comparisons. Report all
seeds, attack counts, clean and attacked accuracy, targeted error,
kidney-left and all-class recall, and loss; do not infer general protection
from an average. Treat the full matrix as descriptive unless an inferential
and multiple-comparison plan is specified in advance. OrganSMNIST remains
exploratory, and its femur-right weakness is a separate clean-quality
limitation, not a second attack. The datasets have different local-epoch
settings, so differences between datasets cannot be attributed solely to
dataset identity.

The [sweep instructions](README.md#organcmnist-and-organsmnist-defense-matrix-sweeps)
describe execution, reference checks, failure handling and the optional
all-dataset launcher. Tests remain unused.

## 0.01 learning-rate pilot

The OrganAMNIST validation-only stability pilot completed all 12 conditions: seeds 42,
49, and 50, each with clean and four-client, 100%-flip runs under FedAvg and
coordinate-wise median. All records report five complete rounds with 20
successful replies, no client failures, finite metrics and model parameters,
and valid runs. All partition manifests verify disjoint, complete assignments;
the four conditions for each seed used the same partition, and the two
attacks selected the same clients. All five recorded round checkpoints and the
final model file are present for each run. Test evaluation was disabled.

The pilot used the CNN, 20 clients, five rounds, one local epoch, batch size 32,
learning rate 0.01, and the kidney-left-to-kidney-right label-flipping attack.
The manifest records a clean working tree at revision
`a2f99fa0ac45def92a24d376fff5a4884f71cb26`; its recorded source hashes match
the current files. Results, partition manifests and execution manifest are in
[`protocol-v3/run-1/`](results/defense/generation-v1/protocol-v3/run-1/);
models and checkpoints are in the matching
[`models/` execution](models/defense/generation-v1/protocol-v3/run-1/).

### 0.01 round-5 validation outcomes

Targeted error is the percentage of kidney-left images predicted as
kidney-right. Kidney-left recall measures correct predictions for that class.
All values are percentages.

| Seed | Aggregation | Clean accuracy | Attacked accuracy | Clean targeted error | Attacked targeted error | Clean recall | Attacked recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | FedAvg | 75.26 | 74.73 | 6.34 | 18.13 | 70.42 | 35.92 |
| 42 | Median | 75.43 | 75.67 | 5.99 | 10.92 | 70.07 | 60.04 |
| 49 | FedAvg | 74.13 | 72.65 | 5.81 | 34.15 | 74.47 | 23.77 |
| 49 | Median | 73.86 | 74.77 | 6.51 | 12.15 | 74.47 | 59.51 |
| 50 | FedAvg | 80.26 | 77.71 | 7.39 | 29.23 | 77.46 | 33.10 |
| 50 | Median | 80.08 | 79.42 | 7.57 | 18.49 | 78.35 | 58.98 |

Mean clean accuracy was 76.55% for FedAvg and 76.45% for median. Seed 49's
median-clean run completed at 0.01, unlike the earlier 0.1 attempt. These three
selected seeds give preliminary evidence, not proof of stability or a general
robustness claim. The 0.01 results are a separate protocol and must not be
pooled with the historical 0.1 results below.

### Frozen OrganAMNIST comparison protocol

The OrganAMNIST validation comparison uses learning rate 0.01, seeds 42–51,
20 clients, five rounds, one local epoch and batch size 32. For each seed,
run FedAvg and median, each clean and with four poisoned clients flipping
100% of kidney-left labels to kidney-right. Run all 40 conditions afresh
from one committed source and captured environment; do not reuse pilot runs.

The primary metric is the per-seed difference in attack-induced targeted
error: `(median attacked - median clean) - (FedAvg attacked - FedAvg clean)`.
Report the mean and each seed's value in percentage points; a negative
difference favors median. Also report each method's paired attack change,
clean accuracy and median-minus-FedAvg clean difference, kidney-left recall,
validation loss, and per-seed outcomes.

The clean-performance floor is **75% mean final-round validation accuracy
separately for each aggregation across all ten seeds**. Compare unrounded
means with the floor; this is a practical criterion, not a stability or
generalization guarantee. Individual seeds may fall below 75%: report them
and class-level recall rather than excluding them. Assess the ten-seed floor
only when all ten clean results for that aggregation are valid, and report an
incomplete comparison separately.

Require all 40 valid conditions for a complete comparison. The runner must
report failures and continue with the remaining conditions. Preserve failed
records and logs; retry a condition at most once only if the failure is
clearly attributable to infrastructure, using the same seed, configuration,
source and environment and retaining a separate execution and its manifest.
Document the reason and both attempts; count a successful retry only once.
Do not silently replace numerical failures or omit seeds. If any condition
remains invalid, report the missing cells and any complete pairs as
exploratory, not as a complete ten-seed comparison. Do not treat an unpaired
condition as a pair.

Seeds 42–51 are also the proposed harmonized set for OrganCMNIST and
OrganSMNIST. Each needs its own stability and clean-performance checks before
adopting settings; this OrganAMNIST pilot does not establish that 0.01 is
suitable for them. Keep test evaluation disabled for the OrganAMNIST
comparison; see the [run instructions](README.md#001-organamnist-comparison).

## Historical 0.1 learning-rate pilot

### Status and protocol

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

### Round-5 validation outcomes

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

### Verification after code reorganization

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
clean seed-49 failure at 0.1 remains unresolved. Keep the three invalid protocol-v3
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
