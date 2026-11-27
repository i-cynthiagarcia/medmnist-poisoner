"""Server aggregation strategies and strict client-reply validation."""

import math

import torch
from flwr.serverapp.strategy import FedAvg


def contains_only_finite_values(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(contains_only_finite_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(contains_only_finite_values(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


class VerifiedFedAvg(FedAvg):
    """FedAvg variant that records per-round client reply counts."""

    def __init__(self, num_clients: int, require_complete_round: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.require_complete_round = require_complete_round
        self.received_by_round = {}
        self.failed_by_round = {}
        self.expected_by_round = {}

    def configure_train(self, server_round, arrays, config, grid):
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if self.require_complete_round:
            self.expected_by_round[server_round] = (
                {msg.metadata.dst_node_id for msg in messages},
                {
                    key: (value.shape, value.dtype)
                    for key, value in arrays.to_torch_state_dict().items()
                },
            )
        return messages

    def _record_and_validate_replies(self, server_round, replies):
        received = sum(1 for reply in replies if not reply.has_error())
        failed = sum(1 for reply in replies if reply.has_error())
        self.received_by_round[server_round] = received
        self.failed_by_round[server_round] = failed
        if not self.require_complete_round:
            return
        if received != self.num_clients or failed:
            raise ValueError(
                f"Round {server_round} requires {self.num_clients} successful "
                f"client replies; received {received} and {failed} failures."
            )
        expected_nodes, expected_arrays = self.expected_by_round[server_round]
        reply_nodes = [reply.metadata.src_node_id for reply in replies]
        if len(set(reply_nodes)) != self.num_clients or set(reply_nodes) != expected_nodes:
            raise ValueError(f"Round {server_round} has duplicate or unexpected client replies.")
        self._check_and_log_replies(replies, is_train=True)
        for reply in replies:
            arrays = reply.content[self.arrayrecord_key].to_torch_state_dict()
            if set(arrays) != set(expected_arrays):
                raise ValueError(f"Round {server_round} has incompatible model keys.")
            for key, value in arrays.items():
                if (value.shape, value.dtype) != expected_arrays[key]:
                    raise ValueError(f"Round {server_round} has incompatible array {key!r}.")
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"Round {server_round} has non-finite array {key!r}.")
            metrics = next(iter(reply.content.metric_records.values()))
            if (
                not contains_only_finite_values(dict(metrics))
                or metrics[self.weighted_by_key] <= 0
            ):
                raise ValueError(f"Round {server_round} has invalid client metrics.")

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        self._record_and_validate_replies(server_round, replies)
        return super().aggregate_train(server_round, replies)


