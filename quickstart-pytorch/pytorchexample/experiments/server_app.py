"""Flower server dispatcher for the configured experiment protocol."""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from pytorchexample.experiments.shared.server import run_server
from pytorchexample.shared.config import require_config_string

app = ServerApp()


def _label_flipping_enabled(context: Context) -> bool:
    experiment_type = require_config_string(
        context.run_config["experiment-type"],
        "experiment-type",
    )
    if experiment_type == "baseline":
        return False
    if experiment_type == "label-flipping":
        return True
    raise ValueError(
        "experiment-type must be either 'baseline' or 'label-flipping'."
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_server(
        grid,
        context,
        label_flipping_enabled=_label_flipping_enabled(context),
    )
