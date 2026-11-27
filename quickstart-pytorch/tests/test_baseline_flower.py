import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from torch.utils.data import DataLoader, TensorDataset

from pytorchexample.experiments import (
    client_app as experiment_client_app,
    server_app as experiment_server_app,
)
from pytorchexample.experiments.shared import client as shared_client
from pytorchexample.shared.model import Net


class BaselineFlowerRegressionTests(unittest.TestCase):
    def test_flower_components_use_experiment_dispatchers(self):
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        contents = project.read_text()
        self.assertIn('clientapp = "pytorchexample.experiments.client_app:app"', contents)
        self.assertIn('serverapp = "pytorchexample.experiments.server_app:app"', contents)

    def test_dispatchers_keep_attack_and_aggregation_independent(self):
        for experiment_type, poisoned in (("baseline", False), ("label-flipping", True)):
            for aggregation in ("fedavg", "median"):
                with self.subTest(experiment_type=experiment_type, aggregation=aggregation):
                    context = Mock()
                    context.run_config = {
                        "experiment-type": experiment_type,
                        "aggregation": aggregation,
                    }
                    message, grid = Mock(), Mock()
                    with (
                        patch.object(experiment_client_app, "train_client") as train,
                        patch.object(experiment_server_app, "run_server") as server,
                    ):
                        experiment_client_app.train(message, context)
                        experiment_server_app.main(grid, context)
                    train.assert_called_once_with(
                        message, context, label_flipping_enabled=poisoned
                    )
                    server.assert_called_once_with(
                        grid, context, label_flipping_enabled=poisoned
                    )

    def test_dispatchers_reject_unknown_experiment_type(self):
        context = Mock()
        context.run_config = {"experiment-type": "overlay"}
        with self.assertRaises(ValueError):
            experiment_client_app.train(Mock(), context)
        with self.assertRaises(ValueError):
            experiment_server_app.main(Mock(), context)

    def test_shared_client_trains_without_poisoning_configuration(self):
        initial_model = Net()
        initial_state = {
            key: value.clone()
            for key, value in initial_model.state_dict().items()
        }
        message = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(initial_state),
                    "config": ConfigRecord({"lr": 0.1}),
                }
            ),
            dst_node_id=1,
            message_type="train",
        )
        context = SimpleNamespace(
            run_config={
                "seed": 42,
                "dataset": "organamnist",
                "batch-size": 2,
                "local-epochs": 2,
            },
            node_config={"partition-id": 1, "num-partitions": 20},
        )
        loader = DataLoader(
            TensorDataset(
                torch.zeros((3, 1, 28, 28)),
                torch.tensor([0, 1, 2]),
            ),
            batch_size=2,
        )

        with (
            patch.object(shared_client, "set_seed"),
            patch.object(
                shared_client,
                "load_data",
                return_value=(loader, None),
            ) as load_data,
            patch.object(shared_client, "train_fn", return_value=0.125) as train_fn,
            patch.object(shared_client.torch.cuda, "is_available", return_value=False),
        ):
            reply = shared_client.train_client(message, context)

        load_data.assert_called_once_with(
            1,
            20,
            2,
            42,
            "organamnist",
            client_dataset_transform=None,
        )
        train_fn.assert_called_once()
        self.assertEqual(train_fn.call_args.args[2:4], (2, 0.1))
        self.assertEqual(reply.content["metrics"]["train_loss"], 0.125)
        self.assertEqual(reply.content["metrics"]["num-examples"], 3)
        returned_state = reply.content["arrays"].to_torch_state_dict()
        for key, value in initial_state.items():
            self.assertTrue(torch.equal(returned_state[key], value))

if __name__ == "__main__":
    unittest.main()
