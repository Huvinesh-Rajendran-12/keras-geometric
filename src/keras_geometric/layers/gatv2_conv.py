from keras import initializers, layers, ops

from .message_passing import MessagePassing


class GATv2Conv(MessagePassing):
    """
    Graph Attention Network v2 (GATv2) Convolution Layer.

    This layer implements the improved Graph Attention Network convolution
    from the paper "How Attentive are Graph Attention Networks?"
    (https://arxiv.org/abs/2105.14491).

    GATv2 addresses a theoretical limitation in the original GAT by using
    a more expressive attention mechanism that allows for dynamic attention.

    Args:
        output_dim (int): Dimensionality of the output features.
        heads (int, optional): Number of multi-head attentions. Defaults to 1.
        concat (bool, optional): Whether to concatenate or average multi-head
            attentions. Defaults to True.
        negative_slope (float, optional): LeakyReLU negative slope. Defaults to 0.2.
        dropout_rate (float, optional): Dropout rate for attention coefficients.
            Defaults to 0.0.
        use_bias (bool, optional): Whether to use bias. Defaults to True.
        kernel_initializer (str, optional): Initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer for bias weights.
            Defaults to 'zeros'.
        add_self_loops (bool, optional): Whether to add self-loops to the
            adjacency matrix. Defaults to True.

    Inherits from MessagePassing layer for graph convolution operations.
    """
    def __init__(self,
                output_dim: int,
                heads: int = 1,
                concat: bool = True,
                negative_slope: float = 0.2,
                dropout_rate: float = 0.0,
                use_bias: bool = True,
                kernel_initializer: str = 'glorot_uniform',
                bias_initializer: str = 'zeros',
                add_self_loops: bool = True,
                **kwargs):
        super().__init__(aggregator='sum', **kwargs)

        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.add_self_loops = add_self_loops

        self.kernel_initializer_obj = initializers.get(self.kernel_initializer)
        self.bias_initializer_obj = initializers.get(self.bias_initializer)

        self.hidden_dim = self.output_dim // self.heads

        if self.output_dim % self.heads != 0:
            raise ValueError(f"Output channels must be divisible by heads, but got {self.output_dim} and {self.heads}")

    def build(self, input_shape):
        """
        Build the layer weights.

        Args:
            input_shape: [(N, F), (2, E)]
        """
        input_dim = input_shape[0]
        if not isinstance(input_dim, (list, tuple)) or len(input_dim) != 2:
            raise ValueError(f"Expected features input shape like (N, F), but got {input_dim}")

        # Linear transformation for node features
        self.linear_transform = layers.Dense(
            self.heads * self.hidden_channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=True
        )

        # Attention weights - the key difference from GAT
        # is that we apply this after linear transform and concatenation
        self.att = self.add_weight(
            shape=(1, self.heads, self.hidden_channels),
            initializer=self.kernel_initializer_obj,
            name="att",
            trainable=True
        )

        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.heads * self.output_dim if self.concat else self.output_dim,),
                initializer=self.bias_initializer_obj,
                name="bias_concat" if self.concat else "bias_average",
                trainable=True
            )
        else:
            self.bias = None

    def call(self, inputs):
        """
        Perform GATv2 convolution.

        Args:
            inputs: List[x, edge_idx]
                - x: [N, F]
                - edge_idx: [2, E]
        Returns:
            [N, output_channels]
        """

        x, edge_idx = inputs
        edge_idx = ops.cast(edge_idx, dtype="int32")
        return self.propagate((x, edge_idx))

    def _compute_attention(self, h_i, h_j, target_idx, num_nodes):
        """
        Forward pass with attention mechanism.

        Args:
            x: [N, heads, output_dim]
            edge_index: [2, E]

        Returns:
            [N, output_channels]
        """
        g_ij = ops.add(h_i, h_j)

        z_ij = ops.leaky_relu(g_ij, negative_slope=self.negative_slope)

        attn_scores = ops.sum(ops.matmul(z_ij, self.attention_weights), axis=-1)

        alpha = self._softmax_by_target(attn_scores, target_idx, num_nodes)

        return ops.expand_dims(alpha, -1) # (E, H, 1)

    def _softmax_by_target(self, scores, target_nodes, num_nodes):
        """
        Compute softmax of attention coefficients, grouped by target nodes.

        Args:
            scores: [E, H]
            target_nodes: [E] Target nodes indices
            num_nodes: Number of nodes

        Returns:
            [E, H] Normalized attention coefficients
        """
        # Create a large negative number for padding
        # min_value = ops.cast(-1e9, dtype=alpha.dtype)

        # We'll compute max and sum for each target node's neighborhood
        target_nodes = ops.cast(target_nodes, dtype='int32')

        # Find max value per target node for numerical stability
        max_per_target = ops.segment_max(scores, target_nodes, num_segments=num_nodes)
        max_per_edge = ops.take(max_per_target, target_nodes, axis=0)

        # Subtract max for numerical stability
        exp_alpha = ops.exp(ops.subtract(scores, max_per_edge))

        # Sum for denominator of softmax
        sum_per_target = ops.segment_sum(exp_alpha, target_nodes, num_segments=num_nodes)
        sum_per_edge = ops.take(sum_per_target, target_nodes, axis=0)

        # Compute softmax
        return ops.divide(exp_alpha, ops.add(sum_per_edge, 1e-10))

    def propagate(self, inputs):
        x, edge_index = inputs
        N = ops.shape(x)[0]
        E = ops.shape(edge_index)[1]

        # linear transformation
        x = self.linear_transform(x)
        x = ops.reshape(x, [N, self.heads, self.hidden_dim])  # [N, heads, hidden_dim]

        source_idx, target_idx = ops.cast(edge_index[0], dtype='int32'), ops.cast(edge_index[1], dtype='int32')
        h_j = ops.take(x, source_idx, axis=0)  # [E, heads, output_dim]

        h_i = ops.take(x, target_idx, axis=0)  # [E, heads, output_dim]

        alpha = self._compute_attention(h_i, h_j, target_idx, N)

        if self.dropout_rate > 0 and self.trainable:
            alpha = layers.Dropout(self.dropout_rate)(alpha)

        # flattening the messages into (E, heads * hidden_dim)
        messages = ops.reshape(self.message(h_i, h_j, alpha=alpha), [E, self.heads * self.hidden_dim])

        aggr_messages = self.aggregate(messages, target_idx, N)

        output = self.update(aggr_messages)

        return output

    def message(self, x_i, x_j, **kwargs):
        """
        Computes messages with attention weights.

        Args:
            x_i: Tensor of shape [E, H, F] containing features of the target nodes.
                Not used in this implementation but required by the MessagePassing interface.
            x_j: Tensor of shape [E, H, F] containing features of the source nodes.
            **kwargs: Contains 'alpha' - the attention coefficients of shape [E, H, 1].

        Returns:
            Tensor of shape [E, H, F] containing the weighted messages.
        """
        alpha = kwargs['alpha']
        return alpha * x_j

    def update(self, aggregated):
        """
        Overrides base update to handle multi-head aggregation and final bias.
        Args:
            aggregated_flat (Tensor): Aggregated messages [N, H * F_out].
        """
        N = ops.shape(aggregated)[0]
        if self.concat:
            output = aggregated
        else:
            aggregated_reshaped = ops.reshape(aggregated, [N, self.heads, self.hidden_dim])
            output = ops.mean(aggregated_reshaped, axis=1)

        if self.use_bias:
            output = ops.add(output, self.bias)

        return output

    def get_config(self):
        """
        Serializes the layer configuration.
        """
        config = super(GATv2Conv, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'heads': self.heads,
            'concat': self.concat,
            'negative_slope': self.negative_slope,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'add_self_loops': self.add_self_loops
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.
        """
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        # Remove 'aggr' and 'aggregator' since 'aggregator' is passed explicitly in __init__
        if 'aggr' in config_copy:
            config_copy.pop('aggr')
        if 'aggregator' in config_copy:
            config_copy.pop('aggregator')
        return cls(**config_copy)
