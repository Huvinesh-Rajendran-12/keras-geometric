from keras import layers, ops

class MessagePassing(layers.Layer):
    def __init__(self, aggr: str = "mean", **kwargs) -> None:
        super(MessagePassing, self).__init__(**kwargs)
        config_aggr = kwargs.pop('aggr', aggr)
        self.aggr = config_aggr
        assert self.aggr in ['mean', 'max', 'sum'], f"Invalid aggregation method: {self.aggr}. Must be one of ['mean', 'max', 'sum']"

    def message(self, x_i, x_j):
        """
        Computes messages from source node j to target node i.
        Default implementation returns x_j, which represents the features of the source node (neighbors).
        Args:
            x_i: Features of the target node [E, F]
            x_j: Features of the source node (neighbours) [E, F]
        Returns:
            Messages to be aggregated [E, F]
        """
        return x_j

    def update(self, aggregated):
        """
        Update node features based on aggregated messages.
        Default implementation returns the aggregated messages.
        Args:
            aggregated: [N, F]
        Returns:
            Updated node features [N, F]
        """
        return aggregated

    def aggregate(self, messages, target_idx, num_nodes):
        """
        Aggregate messages based on target indices.

        Args:
            messages: [E, F]
            target_idx: [E]
        Returns:
        """
        target_idx = ops.cast(target_idx, dtype='int32')
        if self.aggr == 'mean':
            # Compute segment mean by summing and dividing by count
            segment_sum = ops.segment_sum(
                data=messages,
                segment_ids=target_idx,
                num_segments=num_nodes
            )
            segment_count = ops.segment_sum(
                data=ops.ones_like(messages[:, 0:1]),
                segment_ids=target_idx,
                num_segments=num_nodes
            )
            return ops.divide(
                segment_sum,
                ops.maximum(segment_count, ops.ones_like(segment_count))
            )
        elif self.aggr == 'max':
            aggr = ops.segment_max(data=messages, segment_ids=target_idx, num_segments=num_nodes)
            return ops.where(
                ops.isinf(aggr),
                ops.zeros_like(aggr),
                aggr
            )
        elif self.aggr == 'sum':
            return ops.segment_sum(data=messages, segment_ids=target_idx, num_segments=num_nodes)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggr}")

    def propagate(self, inputs):
        """
        Propagate messages through the graph.
        This method is called in the call() method.
        Args:
            inputs: List[x, edge_idx]
                - x: [N, F]
                - edge_idx: [2, E]
        Returns:
            [N, F]
        """
        x, edge_idx = inputs
        N = ops.shape(x)[0]
        E = ops.shape(edge_idx)[1]

        if N == 0:
            F = ops.shape(x)[1] # Number of features
            # Return zero tensor of shape [N, F]
            return ops.zeros((N, F), dtype=x.dtype)  # Return zero tensor of shape [N, F]

        source_node_idx = edge_idx[0]
        target_node_idx = edge_idx[1]
        x_j = ops.take(x, source_node_idx, axis=0)
        x_i = ops.take(x, target_node_idx, axis=0)
        messages = self.message(x_i, x_j)
        aggregated = self.aggregate(messages, target_node_idx, num_nodes=N)
        updates = self.update(aggregated)
        return updates

    def call(self, inputs):
        x, edge_idx = inputs
        edge_idx = ops.cast(edge_idx, dtype='int32')
        return self.propagate([x, edge_idx])

    def get_config(self):
        config = super().get_config()
        config.update({
            'aggr': self.aggr
        })
        return config
