from keras import layers, ops

class MessagePassing(layers.Layer):
    def __init__(self, aggr: str = "mean", **kwargs) -> None:
        super(MessagePassing, self).__init__(**kwargs)
        self.aggr = aggr
        assert self.aggr in ['mean', 'max', 'sum'], f"Invalid aggregation method: {self.aggr}. Must be one of ['mean', 'max', 'sum']"

    def message(self, x_i, x_j):
        """
        Args:
            x_i: Source node features [N, F]
            x_j: Target node features [N, F]
        Returns:
            x_i: [N, F]
        """
        return x_i

    def update(self, aggregated):
        """
        Args:
            aggregated: [N, F]
        Returns:
            [N, F]
        """
        return aggregated

    def aggregate(self, messages, target_idx):
        """
        Args:
            messages: [E, F]
            target_idx: [E]
        Returns:
        """
        if self.aggr == 'mean':
            # Compute segment mean by summing and dividing by count
            segment_sum = ops.segment_sum(messages, target_idx)
            segment_count = ops.segment_sum(ops.ones_like(messages[:, 0:1]), target_idx)
            return ops.divide(segment_sum, ops.maximum(segment_count, ops.ones_like(segment_count)))
        elif self.aggr == 'max':
            return ops.segment_max(messages, target_idx)
        elif self.aggr == 'sum':
            return ops.segment_sum(messages, target_idx)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggr}")

    def propagate(self, inputs):
        """
        Args:
            inputs: List[x, edge_idx]
                - x: [N, F]
                - edge_idx: [2, E]
        Returns:
            [N, F]
        """
        x, edge_idx = inputs
        x_i = ops.take(x, edge_idx[0, :], axis=0)
        x_j = ops.take(x, edge_idx[1, :], axis=0)
        messages = self.message(x_i, x_j)
        aggregated = self.aggregate(messages, edge_idx[1, :])
        updates = self.update(aggregated)
        return updates

    def call(self, inputs):
        x, edge_idx = inputs
        return self.propagate([x, edge_idx])

    def get_config(self):
        config = super(MessagePassing, self).get_config()
        config.update({
            'aggr': self.aggr
        })
        return config
