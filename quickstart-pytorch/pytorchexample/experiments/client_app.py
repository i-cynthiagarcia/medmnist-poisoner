"""Flower client dispatcher for the configured experiment protocol."""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from pytorchexample.experiments.shared.client import (
    evaluate_client,
    train_client,
)
from pytorchexample.shared.config import require_config_string

app = ClientApp()


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


@app.train()
def train(msg: Message, context: Context):
    return train_client(
        msg,
        context,
        label_flipping_enabled=_label_flipping_enabled(context),
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    _label_flipping_enabled(context)
    return evaluate_client(msg, context)
