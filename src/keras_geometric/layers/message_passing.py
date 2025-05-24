from typing import Any

from keras import layers, ops
from keras.src.ops import KerasTensor


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
        aggregator: The aggregation method to use. Must be one of ['mean', 'max', 'sum', 'min', 'std'].
            Defaults to 'mean'.
        **kwargs: Additional arguments passed to the Keras Layer base class.
    """

    def __init__(self, aggregator: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)
        self.aggregator: str = aggregator
        self.supported_aggregators: list[str] = ["mean", "max", "sum", "min", "std"]

        if self.aggregator not in self.supported_aggregators:
            raise ValueError(
                f"Invalid aggregator: {self.aggregator}. "
                f"Must be one of {self.supported_aggregators}"
            )

        # Cache for edge indices to avoid repeated casting
        self._cached_edge_idx: KerasTensor | None = None
        self._cached_edge_idx_hash: int | None = None

        # Additional kwargs that can be passed to message function
        self.message_kwargs: dict[str, any] = {}

    def message(
        self,
        x_i: KerasTensor,
        x_j: KerasTensor,
        edge_attr: KerasTensor | None = None,
        edge_index: KerasTensor | None = None,
        size: tuple[int, int] | None = None,
        **kwargs,
    ) -> KerasTensor:
        """
        Computes messages from source node j to target node i.

        Args:
            x_i: Tensor of shape [E, F] containing features of the target nodes.
                E is the number of edges, F is the number of features.
            x_j: Tensor of shape [E, F] containing features of the source nodes (neighbors).
            edge_attr: Optional tensor of shape [E, D] containing edge attributes.
                D is the number of edge features.
            edge_index: Optional tensor of shape [2, E] containing the edge indices.
            size: Optional tuple (N_i, N_j) indicating the number of target and source nodes.
            **kwargs: Additional arguments that might be used by derived classes.

        Returns:
            Tensor of shape [E, F'] containing the computed messages for each edge.
            F' is the output feature dimension (may differ from F if edge_attr is used).
        """
        if edge_attr is not None:
            # Example: concatenate edge features with source node features
            # Derived classes can override this behavior
            return ops.concatenate([x_j, edge_attr], axis=-1)
        return x_j

    def aggregate(
        self,
        messages: KerasTensor,
        target_idx: KerasTensor,
        num_nodes: int,
        dim_size: int | None = None,
    ) -> KerasTensor:
        """
        Aggregate messages based on target indices using the specified aggregation method.

        Args:
            messages: Tensor of shape [E, F] containing the messages to aggregate.
            target_idx: Tensor of shape [E] containing the target node indices.
            num_nodes: Total number of nodes in the graph.
            dim_size: Optional size of the output dimension (defaults to num_nodes).

        Returns:
            Tensor of shape [N, F] containing the aggregated features for each node.
        """
        if dim_size is None:
            dim_size = num_nodes

        # Handle empty edge case
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1] if len(ops.shape(messages)) > 1 else 1
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        if self.aggregator == "mean":
            # Improved mean aggregation with better numerical stability
            # Use scatter operations for better performance and stability

            # Count the number of messages per node
            ones = ops.ones((ops.shape(messages)[0], 1), dtype=messages.dtype)
            degree = ops.segment_sum(
                data=ones, segment_ids=target_idx, num_segments=dim_size
            )

            # Sum messages per node
            aggregated_sum = ops.segment_sum(
                data=messages, segment_ids=target_idx, num_segments=dim_size
            )

            # Compute mean with numerical stability
            # Add small epsilon to avoid division by zero
            epsilon = ops.convert_to_tensor(1e-8, dtype=degree.dtype)
            degree = ops.maximum(degree, epsilon)

            # Compute mean
            aggregated_mean = aggregated_sum / degree

            # For nodes with no incoming edges, the result should be zeros
            # This is already handled by segment_sum returning zeros
            return aggregated_mean

        elif self.aggregator == "max":
            aggr = ops.segment_max(
                data=messages, segment_ids=target_idx, num_segments=dim_size
            )
            # Replace -inf values with zeros (for nodes with no incoming messages)
            return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)

        elif self.aggregator == "sum":
            return ops.segment_sum(
                data=messages, segment_ids=target_idx, num_segments=dim_size
            )

        elif self.aggregator == "min":
            negative_data = ops.negative(messages)
            aggr = ops.segment_max(
                data=negative_data, segment_ids=target_idx, num_segments=dim_size
            )
            aggr = ops.negative(aggr)
            # Replace inf values with zeros (for nodes with no incoming messages)
            return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)

        elif self.aggregator == "std":
            # Compute standard deviation aggregation using Welford's algorithm for stability
            # First, compute the mean
            ones = ops.ones((ops.shape(messages)[0], 1), dtype=messages.dtype)
            count = ops.segment_sum(
                data=ones, segment_ids=target_idx, num_segments=dim_size
            )

            sum_messages = ops.segment_sum(
                data=messages, segment_ids=target_idx, num_segments=dim_size
            )

            # Safe mean computation
            epsilon = ops.convert_to_tensor(1e-8, dtype=count.dtype)
            safe_count = ops.maximum(count, epsilon)
            mean = sum_messages / safe_count

            # Expand mean to match messages for each edge
            mean_expanded = ops.take(mean, target_idx, axis=0)

            # Compute squared differences - ensure both operands are tensors
            messages_tensor = ops.convert_to_tensor(messages)
            squared_diff = ops.square(messages_tensor - mean_expanded)

            # Sum squared differences per node
            sum_squared_diff = ops.segment_sum(
                data=squared_diff, segment_ids=target_idx, num_segments=dim_size
            )

            # Compute variance (using N instead of N-1 for consistency)
            variance = sum_squared_diff / safe_count

            # Compute standard deviation with numerical stability
            std_dev = ops.sqrt(ops.maximum(variance, ops.zeros_like(variance)))

            # For nodes with single or no neighbors, std should be 0
            std_dev = ops.where(count <= 1, ops.zeros_like(std_dev), std_dev)

            return std_dev

        else:
            raise ValueError(f"Invalid aggregator: {self.aggregator}")

    def update(
        self, aggregated: KerasTensor, x: KerasTensor | None = None
    ) -> KerasTensor:
        """
        Update node features based on aggregated messages.

        Args:
            aggregated: Tensor of shape [N, F] containing the aggregated messages.
            x: Optional tensor of shape [N, F_in] containing the original node features.
                Can be used for residual connections or more complex updates.

        Returns:
            Tensor of shape [N, F_out] containing the updated node features.
        """
        return aggregated

    def pre_aggregate(self, messages: KerasTensor) -> KerasTensor:
        """
        Hook for preprocessing messages before aggregation.
        Can be overridden by derived classes to implement custom preprocessing.

        Args:
            messages: Tensor of shape [E, F] containing messages to preprocess.

        Returns:
            Preprocessed messages of shape [E, F'].
        """
        return messages

    def post_update(self, x: KerasTensor, x_updated: KerasTensor) -> KerasTensor:
        """
        Hook for post-processing after update.
        Can be used for residual connections, normalization, etc.

        Args:
            x: Original node features of shape [N, F_in].
            x_updated: Updated node features of shape [N, F_out].

        Returns:
            Post-processed node features.
        """
        return x_updated

    def propagate(
        self,
        x: KerasTensor | tuple[KerasTensor, KerasTensor],
        edge_index: KerasTensor,
        edge_attr: KerasTensor | None = None,
        size: tuple[int, int] | None = None,
        **kwargs,
    ) -> KerasTensor:
        """
        Propagate messages through the graph by executing the full message passing flow.

        Args:
            x: Tensor of shape [N, F] containing node features, or tuple of tensors
                for bipartite graphs.
            edge_index: Tensor of shape [2, E] containing edge indices.
            edge_attr: Optional tensor of shape [E, D] containing edge attributes.
            size: Optional tuple (N_i, N_j) for bipartite graphs.
            **kwargs: Additional arguments passed to message and update functions.

        Returns:
            Tensor containing the updated node features after message passing.
        """
        # Handle bipartite graphs
        if isinstance(x, (list, tuple)):
            x_i, x_j = x[0], x[1]
            size = (ops.shape(x_i)[0], ops.shape(x_j)[0])
        else:
            x_i = x_j = x
            size = (ops.shape(x)[0], ops.shape(x)[0])

        num_nodes = size[0]  # Number of target nodes

        # Handle empty graph case
        if num_nodes == 0:
            feature_dim = ops.shape(x_i)[1] if len(ops.shape(x_i)) > 1 else 1
            return ops.zeros((0, feature_dim), dtype=x_i.dtype)

        # Check if there are any edges
        num_edges = ops.shape(edge_index)[1]
        if num_edges == 0:
            feature_dim = ops.shape(x_i)[1]
            return ops.zeros((num_nodes, feature_dim), dtype=x_i.dtype)

        # Extract source and target indices
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Gather features for source and target nodes
        x_j_gathered = ops.take(x_j, source_idx, axis=0)
        x_i_gathered = ops.take(x_i, target_idx, axis=0)

        # Compute messages
        messages = self.message(
            x_i_gathered,
            x_j_gathered,
            edge_attr=edge_attr,
            edge_index=edge_index,
            size=size,
            **kwargs,
        )

        # Pre-process messages if needed
        messages = self.pre_aggregate(messages)

        # Aggregate messages
        aggregated = self.aggregate(messages, target_idx, num_nodes, dim_size=size[0])

        # Update node features
        updated = self.update(aggregated, x=x_i)

        # Post-process if needed
        updated = self.post_update(x_i, updated)

        return updated

    # pyrefly: ignore #bad-override
    def call(
        self,
        inputs: list[KerasTensor] | tuple[KerasTensor, ...],
        edge_attr: KerasTensor | None = None,
        training: bool | None = None,
    ) -> KerasTensor:
        """
        Forward pass for the message passing layer.

        Args:
            inputs: List containing [x, edge_index] or tuple of (x, edge_index, edge_attr).
            edge_attr: Optional edge attributes (can also be passed as third element of inputs).
            training: Whether the layer is in training mode.

        Returns:
            Updated node features.
        """
        # Parse inputs
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "Inputs must be a list or tuple containing [x, edge_index]"
            )

        if len(inputs) < 2:
            raise ValueError("Inputs must contain at least [x, edge_index]")

        x = inputs[0]
        edge_index = inputs[1]

        # Edge attributes can be passed as third element of inputs or separately
        if len(inputs) >= 3 and inputs[2] is not None:
            edge_attr = inputs[2]

        # Cast edge_index to int32 and cache if needed
        edge_index_hash = (
            # pyrefly: ignore #implicitly-defined-attribute
            hash(edge_index.ref()) if hasattr(edge_index, "ref") else id(edge_index)
        )
        if (
            self._cached_edge_idx is None
            or self._cached_edge_idx_hash != edge_index_hash
        ):
            self._cached_edge_idx = ops.cast(edge_index, dtype="int32")
            self._cached_edge_idx_hash = edge_index_hash

        edge_index = self._cached_edge_idx

        # Store any additional kwargs for message function
        self.message_kwargs = {}

        return self.propagate(
            x=x, edge_index=edge_index, edge_attr=edge_attr, training=training
        )

    def compute_output_shape(
        self,
        input_shape: list[tuple[int | None, ...]] | tuple[tuple[int | None, ...], ...],
    ) -> tuple[int | None, ...] | tuple[tuple | None, ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape(s) of input tensors.

        Returns:
            Output shape tuple.
        """
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape[0] if len(input_shape) > 0 else input_shape

        # Output shape is same as input node feature shape for base message passing
        return x_shape

    def get_config(self) -> dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({"aggregator": self.aggregator})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MessagePassing":
        """
        Creates a layer from its configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            New layer instance.
        """
        return cls(**config)
