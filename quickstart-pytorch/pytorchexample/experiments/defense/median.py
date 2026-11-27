"""Coordinate-wise median aggregation defense."""

import torch
from flwr.app import ArrayRecord

from pytorchexample.experiments.shared.aggregation import VerifiedFedAvg


class VerifiedMedian(VerifiedFedAvg):
    """Coordinate-wise median with one equally weighted vote per client."""

    def __init__(self, num_clients: int, **kwargs):
        super().__init__(num_clients, require_complete_round=True, **kwargs)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        self._record_and_validate_replies(server_round, replies)
        states = [
            reply.content[self.arrayrecord_key].to_torch_state_dict()
            for reply in replies
        ]
        aggregated = {}
        for key in states[0]:
            values = [state[key] for state in states]
            if not values[0].is_floating_point():
                raise ValueError(f"Median aggregation requires floating-point array {key!r}.")
            ordered = torch.sort(torch.stack(values).to(torch.float64), dim=0).values
            middle = len(values) // 2
            median = ordered[middle]
            if len(values) % 2 == 0:
                median = ordered[middle - 1] / 2 + median / 2
            aggregated[key] = median.to(values[0].dtype)
        metrics = self.train_metrics_aggr_fn(
            [reply.content for reply in replies], self.weighted_by_key
        )
        return ArrayRecord(aggregated), metrics
