"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.shared.config import (
    require_config_float,
    require_config_int,
    require_config_string,
)
from pytorchexample.shared.data import load_data
from pytorchexample.shared.model import Net
from pytorchexample.shared.training import (
    set_seed,
    test as test_fn,
    train as train_fn,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    seed = require_config_int(context.run_config["seed"], "seed")
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    set_seed(seed)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = require_config_int(
        context.node_config["partition-id"], "partition-id"
    )
    num_partitions = require_config_int(
        context.node_config["num-partitions"], "num-partitions"
    )
    batch_size = require_config_int(
        context.run_config["batch-size"], "batch-size"
    )
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, seed, dataset)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        require_config_int(context.run_config["local-epochs"], "local-epochs"),
        require_config_float(msg.content["config"]["lr"], "lr"),
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    seed = require_config_int(context.run_config["seed"], "seed")
    dataset = require_config_string(context.run_config["dataset"], "dataset")
    set_seed(seed)

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = require_config_int(
        context.node_config["partition-id"], "partition-id"
    )
    num_partitions = require_config_int(
        context.node_config["num-partitions"], "num-partitions"
    )
    batch_size = require_config_int(
        context.run_config["batch-size"], "batch-size"
    )
    _, valloader = load_data(partition_id, num_partitions, batch_size, seed, dataset)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
        dataset,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
