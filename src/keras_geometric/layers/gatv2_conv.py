from typing import Any

from keras import KerasTensor, initializers, layers, ops

from keras_geometric.utils import add_self_loops

# Use relative imports to avoid circular import issues
from .message_passing import MessagePassing


class GATv2Conv(MessagePassing):
    """
    Graph Attention Network v2 (GATv2) Convolution Layer.

    Implements the improved Graph Attention Network convolution
    from the paper "How Attentive are Graph Attention Networks?"
    (https://arxiv.org/abs/2105.14491).

    Args:
        output_dim: Dimensionality of the output features per head.
        heads: Number of multi-head attentions. Defaults to 1.
        concat: Whether to concatenate or average multi-head attentions. Defaults to True.
        negative_slope: LeakyReLU negative slope. Defaults to 0.2.
        dropout: Dropout rate for attention coefficients. Defaults to 0.0.
        use_bias: Whether to add bias terms. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        att_initializer: Initializer for attention weights. Defaults to 'glorot_uniform'.
        add_self_loops: Whether to add self-loops. Defaults to True.
        **kwargs: Additional arguments passed to the MessagePassing base class.
    """

    def __init__(
        self,
        output_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        att_initializer: str = "glorot_uniform",
        add_self_loops: bool = True,
        **kwargs,
    ) -> None:
        # GAT uses sum aggregation
        super().__init__(aggregator="sum", **kwargs)

        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.att_initializer = att_initializer
        self.add_self_loops_flag = add_self_loops

        self.features_per_head = output_dim

        # Initialize attributes that will be defined in build
        self.linear_transform: layers.Dense | None = None
        self.att: KerasTensor | None = None
        self.bias: KerasTensor | None = None

    def build(self, input_shape: list[tuple[int, ...]] | tuple[int, ...]) -> None:
        """Build the layer weights."""
        # Extract input shape
        if isinstance(input_shape, list) and len(input_shape) >= 1:
            input_dim_shape = input_shape[0]
        else:
            input_dim_shape = input_shape

        # Handle TensorShape objects
        if hasattr(input_dim_shape, "as_list"):
            input_dim_shape = (
                input_dim_shape.as_list()
            )  # pyrefly: ignore  # missing-attribute
        elif hasattr(input_dim_shape, "__len__"):
            input_dim_shape = tuple(input_dim_shape)

        if not isinstance(input_dim_shape, (list, tuple)) or len(input_dim_shape) != 2:
            raise ValueError(
                f"Expected features input shape like (N, F), but got {input_dim_shape}"
            )

        node_feature_dim = input_dim_shape[1]
        if node_feature_dim is None:
            raise ValueError("Input feature dimension cannot be None.")

        # Linear transformation for node features
        self.linear_transform = layers.Dense(
            self.heads * self.features_per_head,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,  # We handle bias separately
            name="linear_transform",
        )
        self.linear_transform.build((None, node_feature_dim))

        # Attention parameters for GATv2
        self.att = self.add_weight(
            shape=(1, self.heads, self.features_per_head),
            initializer=initializers.get(self.att_initializer),
            name="att",
            trainable=True,
        )

        # Final bias
        if self.use_bias:
            bias_shape = (
                (self.heads * self.features_per_head,)
                if self.concat
                else (self.features_per_head,)
            )
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=initializers.get(self.bias_initializer),
                name="final_bias",
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(
        self,
        inputs: list[KerasTensor] | tuple[KerasTensor, ...],
        edge_attr: KerasTensor | None = None,
        training: bool | None = None,
    ) -> KerasTensor:
        """Perform GATv2 convolution.

        Args:
            inputs: List containing [x, edge_index]
                - x: Node features tensor of shape [N, F]
                - edge_index: Edge indices tensor of shape [2, E]
            edge_attr: Optional edge attributes (not used in GAT)
            training: Boolean indicating training or inference mode

        Returns:
            Tensor of shape [N, output_dim * heads] or [N, output_dim] depending on concat
        """
        # Handle different input formats
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            x, edge_index = inputs[0], inputs[1]
        else:
            raise ValueError(f"Expected inputs to be [x, edge_index], got {inputs}")

        edge_index = ops.convert_to_tensor(edge_index)
        edge_index = ops.cast(edge_index, dtype="int32")

        # Add self-loops if requested
        if self.add_self_loops_flag:
            num_nodes = ops.shape(x)[0]
            edge_index = add_self_loops(edge_index, num_nodes)

        # Use custom propagate method
        return self._gatv2_propagate(x=x, edge_index=edge_index, training=training)

    def propagate(
        self,
        x: KerasTensor | tuple[KerasTensor, KerasTensor],
        edge_index: KerasTensor,
        edge_attr: KerasTensor | None = None,
        size: tuple[int, int] | None = None,
        **kwargs,
    ) -> KerasTensor:
        """Override base class propagate to use GATv2 implementation."""
        training = kwargs.get("training", None)
        return self._gatv2_propagate(x=x, edge_index=edge_index, training=training)

    def _gatv2_propagate(
        self,
        x: KerasTensor | tuple[KerasTensor, KerasTensor],
        edge_index: KerasTensor,
        training: bool | None = None,
    ) -> KerasTensor:
        """Execute the complete GATv2 message passing flow."""
        # Handle bipartite graphs
        if isinstance(x, (list, tuple)):
            x_i, x_j = x[0], x[1]
            size = (ops.shape(x_i)[0], ops.shape(x_j)[0])
        else:
            x_i = x_j = x
            size = (ops.shape(x)[0], ops.shape(x)[0])

        n = size[0]  # Number of target nodes
        e = ops.shape(edge_index)[1]  # Number of edges

        # Handle empty graph case
        if n == 0:
            output_shape = (
                (0, self.heads * self.features_per_head)
                if self.concat
                else (0, self.features_per_head)
            )
            return ops.zeros(output_shape, dtype=x_i.dtype)

        # Handle no edges case
        if e == 0:
            output_shape = (
                (n, self.heads * self.features_per_head)
                if self.concat
                else (n, self.features_per_head)
            )
            return ops.zeros(output_shape, dtype=x_i.dtype)

        # Build layer if not already built
        if self.linear_transform is None or self.att is None:
            # Determine input shape for building
            if isinstance(x, (list, tuple)):
                input_shape = [x[0].shape, x[1].shape]
            else:
                input_shape = x.shape
            self.build(input_shape)

        # Apply linear transformation: [N, F] -> [N, H * F_out]
        if self.linear_transform is None:
            raise RuntimeError("Linear transform layer not built")
        x_transformed = self.linear_transform(x_i)
        # Reshape: [N, H, F_out]
        x_transformed = ops.reshape(
            x_transformed, [n, self.heads, self.features_per_head]
        )

        # Also transform source nodes if bipartite
        if x_i is not x_j:
            if self.linear_transform is None:
                raise RuntimeError("Linear transform layer not built")
            x_j_transformed = self.linear_transform(x_j)
            x_j_transformed = ops.reshape(
                x_j_transformed, [ops.shape(x_j)[0], self.heads, self.features_per_head]
            )
        else:
            x_j_transformed = x_transformed

        source_idx = ops.cast(edge_index[0], dtype="int32")
        target_idx = ops.cast(edge_index[1], dtype="int32")

        # Gather features for each edge: [E, H, F_out]
        h_j = ops.take(x_j_transformed, source_idx, axis=0)
        h_i = ops.take(x_transformed, target_idx, axis=0)

        # Compute attention coefficients: [E, H]
        alpha = self._compute_attention(h_i, h_j, target_idx, n)

        # Apply dropout to attention coefficients if needed
        if self.dropout_layer is not None and training:
            alpha = self.dropout_layer(alpha, training=training)

        # Apply attention to messages: [E, H, F_out]
        # Expand alpha to [E, H, 1] for broadcasting
        alpha_expanded = ops.expand_dims(alpha, -1)
        messages = alpha_expanded * h_j

        # Aggregate messages: [N, H, F_out]
        aggregated = self._aggregate_messages(messages, target_idx, n)

        # Apply final update
        output = self._final_update(aggregated)

        return output

    def _compute_attention(
        self,
        h_i: KerasTensor,
        h_j: KerasTensor,
        target_idx: KerasTensor,
        num_nodes: int,
    ) -> KerasTensor:
        """Compute attention coefficients for each edge."""
        # GATv2: attention(h_i, h_j) = a^T * LeakyReLU(h_i + h_j)
        g_ij = ops.add(h_i, h_j)  # [E, H, F_out]
        z_ij = ops.leaky_relu(g_ij, negative_slope=self.negative_slope)

        if self.att is None:
            raise RuntimeError("Attention weights not built.")

        # Compute attention scores: [E, H]
        attn_scores = ops.sum(ops.multiply(z_ij, self.att), axis=-1)

        # Apply softmax grouped by target nodes
        alpha = self._softmax_by_target(attn_scores, target_idx, num_nodes)

        return alpha

    def _softmax_by_target(
        self, scores: KerasTensor, target_nodes: KerasTensor, num_nodes: int
    ) -> KerasTensor:
        """Compute softmax of attention coefficients, grouped by target nodes."""
        target_nodes = ops.cast(target_nodes, dtype="int32")

        # Find max score per target node for numerical stability
        max_per_target = ops.segment_max(scores, target_nodes, num_segments=num_nodes)
        max_per_edge = ops.take(max_per_target, target_nodes, axis=0)

        # Compute exp(score - max)
        exp_scores = ops.exp(ops.subtract(scores, max_per_edge))

        # Sum exp scores per target node
        sum_per_target = ops.segment_sum(
            exp_scores, target_nodes, num_segments=num_nodes
        )
        sum_per_edge = ops.take(sum_per_target, target_nodes, axis=0)

        # Compute softmax
        return ops.divide(exp_scores, ops.add(sum_per_edge, 1e-10))

    def _aggregate_messages(
        self, messages: KerasTensor, target_idx: KerasTensor, num_nodes: int
    ) -> KerasTensor:
        """Aggregate messages for each target node."""
        # messages: [E, H, F_out]
        # target_idx: [E]
        # Returns: [N, H, F_out]

        # Reshape messages to [E, H*F_out] for segment operations
        e = ops.shape(messages)[0]
        messages_flat = ops.reshape(messages, [e, self.heads * self.features_per_head])

        # Aggregate using segment_sum
        aggregated_flat = ops.segment_sum(
            messages_flat, target_idx, num_segments=num_nodes
        )

        # Reshape back to [N, H, F_out]
        aggregated = ops.reshape(
            aggregated_flat, [num_nodes, self.heads, self.features_per_head]
        )

        return aggregated

    def _final_update(self, aggregated: KerasTensor) -> KerasTensor:
        """Final update step: handle multi-head outputs and apply bias."""
        n = ops.shape(aggregated)[0]

        if self.concat:
            # Keep concatenated features: [N, H*F_out]
            output = ops.reshape(aggregated, [n, self.heads * self.features_per_head])
        else:
            # Average across heads: [N, F_out]
            output = ops.mean(aggregated, axis=1)

        # Add final bias
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        return output

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
        Default message function for compatibility with MessagePassing base class.
        This is not used in the main GATv2 flow but may be called by tests.
        """
        return x_j

    def get_config(self) -> dict[str, Any]:
        """Serialize the layer configuration."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "heads": self.heads,
                "concat": self.concat,
                "negative_slope": self.negative_slope,
                "dropout": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "att_initializer": self.att_initializer,
                "add_self_loops": self.add_self_loops_flag,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GATv2Conv":
        """Create a layer from its config."""
        # Handle potential naming differences
        if "add_self_loops" in config and "add_self_loops_flag" not in config:
            config["add_self_loops"] = config.get("add_self_loops", True)

        # Remove aggregator from config since we set it explicitly in __init__
        config = dict(config)  # Make a copy to avoid modifying original
        config.pop("aggregator", None)

        return cls(**config)
