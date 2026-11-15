"""Shared naming helpers for generations and run-specific artifacts."""

import re
from pathlib import Path


def run_artifact_stem(seed: int, run_id: int | str) -> str:
    """Return a stable, collision-resistant stem for one Flower run."""
    return f"seed-{seed}_run-{run_id}"


def resolve_protocol_id(configured_id: str) -> str:
    """Resolve and validate the stable protocol version ID."""
    if configured_id == "auto":
        return "protocol-v1"
    if not re.fullmatch(r"protocol-v\d+", configured_id):
        raise ValueError("protocol-id must use the form 'protocol-vN'.")
    return configured_id


def resolve_execution_id(
    configured_id: str,
    results_root: str | Path,
    models_root: str | Path,
    generation_id: str,
    protocol_id: str,
) -> str:
    """Resolve an explicit run ID or choose the next available run number."""
    if configured_id != "auto":
        if not re.fullmatch(r"run-\d+", configured_id):
            raise ValueError(
                "execution-id must be 'auto' or use the form 'run-N'."
            )
        return configured_id

    highest_run = 0
    for root in (results_root, models_root):
        protocol_dir = Path(root) / generation_id / protocol_id
        if not protocol_dir.is_dir():
            continue
        for path in protocol_dir.iterdir():
            match = re.fullmatch(r"run-(\d+)", path.name)
            if match:
                highest_run = max(highest_run, int(match.group(1)))
    return f"run-{highest_run + 1}"


def validate_generation_id(generation_id: str) -> str:
    """Validate a generation directory name."""
    if not re.fullmatch(r"generation-v\d+", generation_id):
        raise ValueError("generation-id must use the form 'generation-vN'.")
    return generation_id
