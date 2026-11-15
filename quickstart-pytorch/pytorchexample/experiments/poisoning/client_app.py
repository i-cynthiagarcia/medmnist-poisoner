"""Flower client entry point for label-flipping poisoning runs."""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from pytorchexample.experiments.shared.client import (
    evaluate_client,
    train_client,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    return train_client(msg, context, label_flipping_enabled=True)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    return evaluate_client(msg, context)
