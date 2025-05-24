from typing import Any

from keras import initializers, layers, ops

# Use relative imports to avoid circular import issues
from .message_passing import MessagePassing


# Helper function to add self-loops
def add_self_loops(edge_index: Any, num_nodes: int) -> Any:
    """Adds self-loops to edge_index using keras.ops."""
    # Ensure edge_index has shape (2, E)
    if ops.shape(edge_index)[0] != 2:
        edge_index = ops.stack([edge_index[0], edge_index[1]], axis=0)
    loop_indices = ops.arange(0, num_nodes, dtype=edge_index.dtype)
    self_loops = ops.stack([loop_indices, loop_indices], axis=0)
    edge_index_with_loops = ops.concatenate([edge_index, self_loops], axis=1)
    return edge_index_with_loops


class GATv2Conv(MessagePassing):
    """
    Graph Attention Network v2 (GATv2) Convolution Layer.

    Implements the improved Graph Attention Network convolution
    from the paper "How Attentive are Graph Attention Networks?"
    (https://arxiv.org/abs/2105.14491).

    Args:
        output_dim (int): Dimensionality of the output features per head.
        heads (int, optional): Number of multi-head attentions. Defaults to 1.
        concat (bool, optional): Whether to concatenate or average multi-head attentions. Defaults to True.
        negative_slope (float, optional): LeakyReLU negative slope. Defaults to 0.2.
        dropout (float, optional): Dropout rate for attention coefficients. Defaults to 0.0.
        use_bias (bool, optional): Whether to add bias terms. Defaults to True.
        kernel_initializer (str, optional): Initializer for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer for bias weights. Defaults to 'zeros'.
        att_initializer (str, optional): Initializer for attention weights. Defaults to 'glorot_uniform'.
        add_self_loops (bool, optional): Whether to add self-loops. Defaults to True.
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
        **kwargs: Any,
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
        self.add_self_loops = add_self_loops

        self.features_per_head = output_dim

        # Initialize attributes that will be defined in build
        self.linear_transform = None
        self.att = None
        self.bias = None
        self.training = False  # Initialize training flag to satisfy Pyrefly

    def build(self, input_shape: Any) -> None:
        """Build the layer weights."""
        # Extract input shape
        if isinstance(input_shape, list) and len(input_shape) >= 1:
            input_dim_shape = input_shape[0]
        else:
            input_dim_shape = input_shape

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
            use_bias=self.use_bias,
            bias_initializer=self.bias_initializer,
            name="linear_transform",
        )
        self.linear_transform.build((None, node_feature_dim))

        # Attention parameters - using original GAT mechanism for now
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

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Perform GATv2 convolution.

        Args:
            inputs: List containing [x, edge_index]
                - x: Node features tensor of shape [N, F]
                - edge_index: Edge indices tensor of shape [2, E]
            training: Boolean indicating training or inference mode

        Returns:
            Tensor of shape [N, output_dim * heads] or [N, output_dim] depending on concat
        """
        # Handle different input formats
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, edge_index = inputs
        else:
            raise ValueError(f"Expected inputs to be [x, edge_index], got {inputs}")

        edge_index = ops.cast(edge_index, dtype="int32")

        # Add self-loops if requested
        if self.add_self_loops:
            num_nodes = ops.shape(x)[0]
            edge_index = add_self_loops(edge_index, num_nodes)

        # Pass data and training flag to propagate
        return self.propagate(inputs=(x, edge_index), training=training)

    def _compute_attention(self, h_i, h_j, target_idx, num_nodes):
        """Compute attention coefficients for each edge."""
        g_ij = ops.add(h_i, h_j)
        z_ij = ops.leaky_relu(g_ij, negative_slope=self.negative_slope)
        if self.att is None:
            raise RuntimeError("Attention weights not built.")
        attn_scores = ops.sum(ops.multiply(z_ij, self.att), axis=-1)
        alpha = self._softmax_by_target(attn_scores, target_idx, num_nodes)
        return ops.expand_dims(alpha, -1)

    def _softmax_by_target(self, scores, target_nodes, num_nodes):
        """Compute softmax of attention coefficients, grouped by target nodes."""
        target_nodes = ops.cast(target_nodes, dtype="int32")
        max_per_target = ops.segment_max(scores, target_nodes, num_segments=num_nodes)

        max_per_edge = ops.take(max_per_target, target_nodes, axis=0)
        exp_alpha = ops.exp(ops.subtract(scores, max_per_edge))

        sum_per_target = ops.segment_sum(
            exp_alpha, target_nodes, num_segments=num_nodes
        )
        current_sum_size = ops.shape(sum_per_target)[0]
        if current_sum_size < num_nodes:
            padding_sum = ops.zeros(
                (num_nodes - current_sum_size, ops.shape(scores)[1]), dtype=scores.dtype
            )
            sum_per_target = ops.concatenate([sum_per_target, padding_sum], axis=0)

        sum_per_edge = ops.take(sum_per_target, target_nodes, axis=0)
        return ops.divide(exp_alpha, ops.add(sum_per_edge, 1e-10))

    def propagate(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute the complete GATv2 message passing flow.

        Args:
            inputs: Tuple containing (x, edge_index)
                - x: Node features tensor of shape [N, F]
                - edge_index: Edge indices tensor of shape [2, E]
            **kwargs: Additional arguments, including training flag

        Returns:
            Tensor of shape [N, output_dim * heads] or [N, output_dim] depending on concat
        """
        # Extract node features, edge indices, and training flag
        x, edge_index = inputs
        training = kwargs.get("training", None)
        n = ops.shape(x)[0]
        e = ops.shape(edge_index)[1]

        # Handle empty graph case
        if n == 0:
            output_shape = (
                (0, self.heads * self.features_per_head)
                if self.concat
                else (0, self.features_per_head)
            )
            return ops.zeros(output_shape, dtype=x.dtype)

        # Handle no edges case
        if e == 0:
            output_shape = (
                (n, self.heads * self.features_per_head)
                if self.concat
                else (n, self.features_per_head)
            )
            return ops.zeros(output_shape, dtype=x.dtype)

        if self.linear_transform is None or self.att is None:
            raise RuntimeError("Layer weights not built. Call layer on data first.")

        # Apply linear transformation: [N, F] -> [N, H * F_out]
        x_transformed = self.linear_transform(x)
        # Reshape: [N, H, F_out]
        x_transformed = ops.reshape(
            x_transformed, [n, self.heads, self.features_per_head]
        )

        source_idx = ops.cast(edge_index[0], dtype="int32")
        target_idx = ops.cast(edge_index[1], dtype="int32")

        h_j = ops.take(x_transformed, source_idx, axis=0)
        h_i = ops.take(x_transformed, target_idx, axis=0)

        # Compute attention coefficients: [E, H, 1]
        alpha = self._compute_attention(h_i, h_j, target_idx, n)

        # Apply dropout to attention coefficients if dropout rate > 0
        if self.dropout_layer is not None:
            alpha = self.dropout_layer(alpha, training=training)

        # Compute messages (apply attention): [E, H, F_out]
        messages = self.message(h_i, h_j, alpha=alpha)

        # Aggregate messages (using base class 'sum' aggregation)
        messages_flat = ops.reshape(messages, [e, self.heads * self.features_per_head])
        aggregated = super().aggregate(messages_flat, target_idx, num_nodes=n)

        # Final update (concat/average + final bias)
        output = self.update(aggregated)

        return output

    def message(self, x_i: Any, x_j: Any, **kwargs: Any) -> Any:
        """Computes messages with attention weights."""
        alpha = kwargs["alpha"]  # [E, H, 1]
        return alpha * x_j  # [E, H, F_out]

    def update(self, aggregated: Any) -> Any:
        """Final update step: handle multi-head outputs and apply bias."""
        n = ops.shape(aggregated)[0]

        if self.concat:
            # Keep concatenated features
            output = aggregated  # [N, H*F_out]
        else:
            # Average across heads
            aggregated_reshaped = ops.reshape(
                aggregated, [n, self.heads, self.features_per_head]
            )
            output = ops.mean(aggregated_reshaped, axis=1)  # [N, F_out]

        # Add final bias
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        return output

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
                "add_self_loops": self.add_self_loops,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GATv2Conv":
        """Create a layer from its config."""
        config.pop("aggr", None)  # Remove aggregator if present
        return cls(**config)
