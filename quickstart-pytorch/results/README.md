# Results Index

Results are grouped by experiment category, then by generation:

```text
results/<category>/generation-vN/protocol-vN/run-N/<dataset>/<experiment>/
```

The supported categories are `baseline`, `poisoning`, and `defense`.
Generation numbers describe successive eras within a category; they do not
assert that every result in a generation is directly comparable. `protocol-vN`
identifies a protocol structure, and `run-N` identifies one execution of that
protocol. Reuse the protocol version for reruns with the same structure and
advance the run number; use a new protocol version when the structure changes.
Migrated timestamp-based records preserve their former ID in
`original_protocol_id`. Run JSON names also include the seed and Flower run
ID. Model checkpoints use the matching layout under `models/`.

## Baseline generations

- **generation-v0** — archived early and failed records, retained as project
  history. Protocol and environment details may be incomplete.
- **generation-v1** — earlier per-dataset baseline records. Their environments
  and procedures are not consistently verified; do not combine them with
  later results for selection or direct comparison.
- **generation-v2** — the five-seed Validation-v2 baseline. Its final saved
  checkpoints have post-hoc per-image validation metrics, but original
  per-round loss and client-training records are unchanged. See
  [baseline_results.md](../baseline_results.md) for status and interpretation.
- **generation-v3** — the next corrected baseline era. The default script mode
  records one-round runs for a quick check; use `--full-sweep` for the
  seven-setting, five-seed, three-dataset baseline matrix, or
  `--selected-settings` to reproduce the current candidate controls across
  five seeds. Both modes record per-image losses, per-round validation
  metrics, confusion matrices, plots, and global checkpoints. After a full
  sweep, run `python summarize_baseline_sweep.py` from `quickstart-pytorch/`
  to validate its records and print a Markdown summary. Use `--output <path>`
  to save the report alongside that execution. The summarizer does not modify
  `baseline_results.md`.

## Poisoning results

See [poisoning_results.md](../poisoning_results.md) for the current
label-flipping trial, verification status, paired clean comparison, and
refinement protocol. Run-specific JSON records remain the source of truth for
full metrics and artifact paths.

Generation numbers are local to each category. For example, a future
`poisoning/generation-v0` would be the first poisoning-results era, not the
same protocol as `baseline/generation-v0`.

PyTorch checkpoints under `quickstart-pytorch/models/` are tracked with Git
LFS. Collaborators need Git LFS installed and repository read access to fetch
the checkpoint contents. The LFS pointer and `.gitattributes` rules are stored
in Git; model binaries become available to collaborators after those files are
added, committed, and pushed successfully.

Install Git LFS before cloning, or run `git lfs pull` after installing it in an
existing clone:

```bash
git lfs install
git clone https://github.com/i-cynthiagarcia/medmnist-poisoner.git
```
