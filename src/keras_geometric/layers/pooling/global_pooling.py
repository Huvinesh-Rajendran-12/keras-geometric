"""Global pooling operations for graph-level representations."""

from typing import Any, Literal

import keras
from keras import layers, ops


class GlobalPooling(layers.Layer):
    """
    Global pooling layer for creating graph-level representations.

    This layer performs global pooling over all nodes in a graph to create
    a single graph-level representation. Supports different pooling operations
    including mean, max, sum, and attention-based pooling.

    Args:
        pooling: The pooling operation to use. One of:
            - "mean": Global mean pooling
            - "max": Global max pooling
            - "sum": Global sum pooling
        **kwargs: Additional arguments passed to the base Layer class.

    Example:
        ```python
        import keras
        import numpy as np
        from keras_geometric.layers.pooling import GlobalPooling

        # Create sample node features
        node_features = keras.ops.convert_to_tensor(
            np.random.randn(100, 64), dtype="float32"
        )

        # Create pooling layer
        pool = GlobalPooling(pooling="mean")

        # Get graph-level representation
        graph_repr = pool(node_features)  # Shape: (1, 64)
        ```
    """

    def __init__(
        self,
        pooling: Literal["mean", "max", "sum"] = "mean",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if pooling not in ["mean", "max", "sum"]:
            raise ValueError(
                f"pooling must be one of ['mean', 'max', 'sum'], got {pooling}"
            )

        self.pooling = pooling

    def call(
        self, inputs: keras.KerasTensor, **kwargs
    ) -> keras.KerasTensor:  # pyrefly: ignore  # bad-override
        """
        Apply global pooling to node features.

        Args:
            inputs: Node features tensor of shape [num_nodes, num_features]

        Returns:
            Graph-level representation tensor of shape [1, num_features]
        """
        # Apply the specified pooling operation
        if self.pooling == "mean":
            # Global mean pooling
            pooled = ops.mean(inputs, axis=0, keepdims=True)
        elif self.pooling == "max":
            # Global max pooling
            pooled = ops.max(inputs, axis=0, keepdims=True)
        elif self.pooling == "sum":
            # Global sum pooling
            pooled = ops.sum(inputs, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape given input shape."""
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected input shape to be 2D (num_nodes, num_features), "
                f"got {len(input_shape)}D"
            )

        # Output shape is (1, num_features) - single graph representation
        return (1, input_shape[1])

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({"pooling": self.pooling})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlobalPooling":
        """Create layer from configuration."""
        return cls(**config)


class BatchGlobalPooling(layers.Layer):
    """
    Global pooling layer for batched graphs.

    This layer performs global pooling over nodes in each graph of a batch,
    creating graph-level representations for multiple graphs simultaneously.

    Args:
        pooling: The pooling operation to use. One of:
            - "mean": Global mean pooling
            - "max": Global max pooling
            - "sum": Global sum pooling
        **kwargs: Additional arguments passed to the base Layer class.

    Example:
        ```python
        import keras
        import numpy as np
        from keras_geometric.layers.pooling import BatchGlobalPooling

        # Create sample batched node features and batch indices
        node_features = keras.ops.convert_to_tensor(
            np.random.randn(200, 64), dtype="float32"
        )
        # batch[i] indicates which graph node i belongs to
        batch = keras.ops.convert_to_tensor(
            np.repeat([0, 1, 2], [50, 75, 75]), dtype="int32"
        )

        # Create pooling layer
        pool = BatchGlobalPooling(pooling="mean")

        # Get graph-level representations
        graph_reprs = pool([node_features, batch])  # Shape: (3, 64)
        ```
    """

    def __init__(
        self,
        pooling: Literal["mean", "max", "sum"] = "mean",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if pooling not in ["mean", "max", "sum"]:
            raise ValueError(
                f"pooling must be one of ['mean', 'max', 'sum'], got {pooling}"
            )

        self.pooling = pooling

    def call(  # pyrefly: ignore  # bad-override
        self, inputs: list[keras.KerasTensor] | tuple[keras.KerasTensor, ...], **kwargs
    ) -> keras.KerasTensor:
        """
        Apply global pooling to batched node features.

        Args:
            inputs: List/tuple containing:
                - node_features: Node features tensor of shape [total_nodes, num_features]
                - batch: Batch indices tensor of shape [total_nodes] indicating
                        which graph each node belongs to

        Returns:
            Graph-level representations tensor of shape [num_graphs, num_features]
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "inputs must be a list/tuple of [node_features, batch], "
                f"got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        node_features, batch = inputs

        # Get the number of graphs in the batch
        num_graphs = ops.cast(ops.max(batch) + 1, dtype="int32")
        num_features = ops.shape(node_features)[1]

        # Initialize output tensor
        if self.pooling == "max":
            # For max pooling, initialize with negative infinity
            output = ops.full(
                (num_graphs, num_features), float("-inf"), dtype=node_features.dtype
            )
        else:
            # For mean and sum pooling, initialize with zeros
            output = ops.zeros((num_graphs, num_features), dtype=node_features.dtype)

        # Perform pooling for each graph in the batch
        output_list = []
        for graph_id in range(num_graphs):
            # Find nodes belonging to this graph
            mask = ops.equal(batch, graph_id)

            # Get indices where mask is True
            indices = ops.where(mask)

            # Gather features for this graph
            if ops.shape(indices)[0] > 0:  # Check if graph has nodes
                # ops.where returns [N, 1] shape, so squeeze to get [N]
                node_indices = ops.reshape(indices, (-1,))
                graph_nodes = ops.take(node_features, node_indices, axis=0)

                # Apply pooling operation
                if self.pooling == "mean":
                    pooled = ops.mean(graph_nodes, axis=0)
                elif self.pooling == "max":
                    pooled = ops.max(graph_nodes, axis=0)
                elif self.pooling == "sum":
                    pooled = ops.sum(graph_nodes, axis=0)
                else:
                    raise ValueError(f"Unknown pooling type: {self.pooling}")
            else:
                # Empty graph - return zeros
                pooled = ops.zeros((num_features,), dtype=node_features.dtype)

            output_list.append(pooled)

        # Stack results
        output = ops.stack(output_list, axis=0)

        return output

    def compute_output_shape(
        self, input_shape: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]
    ) -> tuple[int | None, int]:
        """Compute the output shape given input shapes."""
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "input_shape must be a list/tuple of 2 shapes for [node_features, batch]"
            )

        node_features_shape, batch_shape = input_shape

        # Handle the case where input_shape is a single tuple (incorrect usage)
        if isinstance(node_features_shape, int):
            raise ValueError(
                "input_shape must be a list/tuple of 2 shapes for [node_features, batch], "
                f"got single shape {input_shape}"
            )

        if len(node_features_shape) != 2:
            raise ValueError(
                f"Expected node_features shape to be 2D (total_nodes, num_features), "
                f"got {len(node_features_shape)}D"
            )

        if len(batch_shape) != 1:
            raise ValueError(
                f"Expected batch shape to be 1D (total_nodes,), got {len(batch_shape)}D"
            )

        # Output shape is (num_graphs, num_features)
        # num_graphs is dynamic based on batch content
        return (None, node_features_shape[1])

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({"pooling": self.pooling})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BatchGlobalPooling":
        """Create layer from configuration."""
        return cls(**config)
