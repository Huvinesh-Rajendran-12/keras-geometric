from typing import Any

from keras import layers, ops


class MessagePassing(layers.Layer):
    """
    Base class for all message passing graph neural network layers.

    This class implements the general message passing framework that consists of three steps:
    1. Message computation: Compute messages between connected nodes
    2. Aggregation: Aggregate messages from neighbors for each node
    3. Update: Update node features based on aggregated messages

    Derived classes can customize these steps by overriding the `message`, `aggregate`,
    and `update` methods.

    Args:
        aggregator: The aggregation method to use. Must be one of ['mean', 'max', 'sum', 'pooling'].
            Defaults to 'mean'.
        **kwargs: Additional arguments passed to the Keras Layer base class.
    """

    def __init__(self, aggregator: str = "mean", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.aggregator = aggregator
        assert (
            self.aggregator in ["mean", "max", "sum", "pooling"]
        ), f"Invalid aggregator: {self.aggregator}. Must be one of ['mean', 'max', 'sum', 'pooling']"

    def message(self, x_i: Any, x_j: Any, **kwargs: Any) -> Any:
        """
        Computes messages from source node j to target node i.
        Default implementation returns x_j, which represents the features of the source node (neighbors).

        Args:
            x_i: Tensor of shape [E, F] containing features of the target nodes.
                E is the number of edges, F is the number of features.
                Not used in the default implementation but available for derived classes.
            x_j: Tensor of shape [E, F] containing features of the source nodes (neighbors).
            **kwargs: Additional arguments that might be used by derived classes.

        Returns:
            Tensor of shape [E, F] containing the computed messages for each edge.
        """
        return x_j

    def update(self, aggregated: Any) -> Any:
        """
        Update node features based on aggregated messages.
        Default implementation returns the aggregated messages without modification.

        Args:
            aggregated: Tensor of shape [N, F] containing the aggregated messages for each node.
                N is the number of nodes, F is the number of features.

        Returns:
            Tensor of shape [N, F] containing the updated node features.
        """
        return aggregated

    def aggregate(self, messages: Any, target_idx: Any, num_nodes: int) -> Any:
        """
        Aggregate messages based on target indices using the specified aggregation method.
        Handles different aggregation types (mean, max, sum) and special cases like empty graphs.

        Args:
            messages: Tensor of shape [E, F] containing the messages to aggregate.
                E is the number of edges, F is the number of features.
            target_idx: Tensor of shape [E] containing the target node indices for each message.
            num_nodes: Integer specifying the total number of nodes in the graph.

        Returns:
            Tensor of shape [N, F] containing the aggregated features for each node.
            N is the number of nodes, F is the number of features.

        Raises:
            ValueError: If the specified aggregation method is not supported.
        """
        # Handle empty edge case - if there are no edges, return zeros
        if ops.shape(messages)[0] == 0:
            # For empty graphs, we need to know the feature dimension
            # If messages is empty, we can still access its shape
            if hasattr(messages, "shape") and len(messages.shape) > 1:
                feature_dim = messages.shape[1]
                return ops.zeros((num_nodes, feature_dim), dtype=messages.dtype)
            else:
                # Fallback - just return zeros matching the input node features
                return ops.zeros(
                    (num_nodes, ops.shape(messages)[1]), dtype=messages.dtype
                )

        target_idx = ops.cast(target_idx, dtype="int32")
        if self.aggregator == "mean":
            # Compute segment mean by summing and dividing by count
            segment_sum = ops.segment_sum(
                data=messages, segment_ids=target_idx, num_segments=num_nodes
            )
            segment_count = ops.segment_sum(
                data=ops.ones_like(messages[:, 0:1]),
                segment_ids=target_idx,
                num_segments=num_nodes,
            )
            return ops.divide(
                segment_sum, ops.maximum(segment_count, ops.ones_like(segment_count))
            )
        elif self.aggregator == "max":
            aggr = ops.segment_max(
                data=messages, segment_ids=target_idx, num_segments=num_nodes
            )
            return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)
        elif self.aggregator == "sum":
            return ops.segment_sum(
                data=messages, segment_ids=target_idx, num_segments=num_nodes
            )
        elif self.aggregator == "pooling":
            # For pooling, we use max aggregation (assuming messages have been pre-processed by MLP)
            aggr = ops.segment_max(
                data=messages, segment_ids=target_idx, num_segments=num_nodes
            )
            return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)
        else:
            raise ValueError(f"Invalid aggregator: {self.aggregator}")

    def propagate(self, inputs: Any) -> Any:
        """
        Propagate messages through the graph by executing the full message passing flow:
        1. Extract node features and edge indices
        2. Compute messages between connected nodes
        3. Aggregate messages for each target node
        4. Update node features based on aggregated messages

        This method is called by the `call()` method and orchestrates the message passing process.

        Args:
            inputs: List containing [x, edge_idx]
                - x: Tensor of shape [N, F] containing node features.
                  N is the number of nodes, F is the number of features.
                - edge_idx: Tensor of shape [2, E] containing edge indices as [source_nodes, target_nodes].
                  E is the number of edges.

        Returns:
            Tensor of shape [N, F] containing the updated node features after message passing.
        """
        x, edge_idx = inputs
        n = ops.shape(x)[0]  # Number of nodes
        # If there are no nodes, return empty tensor
        if n == 0:
            f = ops.shape(x)[1]  # Number of features
            return ops.zeros((n, f), dtype=x.dtype)

        # Check if there are any edges
        e = ops.shape(edge_idx)[1]  # Number of edges
        if e == 0:
            # No edges case - return zeros for all nodes
            f = ops.shape(x)[1]  # Number of features
            return ops.zeros((n, f), dtype=x.dtype)

        # If there are edges, perform normal message passing
        source_node_idx = edge_idx[0]
        target_node_idx = edge_idx[1]
        x_j = ops.take(x, source_node_idx, axis=0)
        x_i = ops.take(x, target_node_idx, axis=0)
        messages = self.message(x_i, x_j)
        aggregated = self.aggregate(messages, target_node_idx, num_nodes=n)
        updates = self.update(aggregated)
        return updates

    # pyrefly: ignore # implicityly-defined-attribute
    def call(
        self, inputs: Any, training: bool | None = None
    ) -> Any:  # pyrefly: ignore  # bad-override
        """
        Forward pass for the message passing layer.

        Args:
            inputs: List or tuple containing [x, edge_idx]
                - x: Tensor of shape [N, F] containing node features
                - edge_idx: Tensor of shape [2, E] containing edge indices

        Returns:
            Tensor of shape [N, F] containing the updated node features
        """
        # Get the inputs from args or kwargs
        x, edge_idx = inputs
        edge_idx = ops.cast(edge_idx, dtype="int32")
        return self.propagate([x, edge_idx])

    def get_config(self) -> dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({"aggregator": self.aggregator})
        return config
