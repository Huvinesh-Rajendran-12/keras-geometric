from typing import Any, Optional, Tuple

import keras
import numpy as np  # For -np.inf
from keras import activations, initializers, layers, ops

# Assuming MessagePassing is in the same directory
from .message_passing import MessagePassing

# Assuming utils are in sibling directory 'utils' relative to package root
try:
    from keras_geometric.utils.main import add_self_loops
except ImportError:
    # Fallback: Define inline if utils import fails (useful for testing standalone)
    print("Warning: Could not import add_self_loops from utils. Using inline definition.")
    def add_self_loops(edge_index, num_nodes):
        loop_indices = ops.arange(0, num_nodes, dtype=edge_index.dtype)
        self_loops = ops.stack([loop_indices, loop_indices], axis=0)
        return ops.concatenate([edge_index, self_loops], axis=1)


class GATv2Conv(MessagePassing):
    """
    Graph Attention Network v2 (GATv2) Convolution Layer.

    Implements the improved Graph Attention Network convolution
    from the paper "How Attentive are Graph Attention Networks?"
    (https://arxiv.org/abs/2105.14491).

    Bias handling aims to match PyTorch Geometric: bias is applied in the
    initial linear transform AND as a final additive term if use_bias=True.

    Args:
        output_dim (int): Dimensionality of the output features per head.
        heads (int, optional): Number of multi-head attentions. Defaults to 1.
        concat (bool, optional): Whether to concatenate or average multi-head attentions. Defaults to True.
        negative_slope (float, optional): LeakyReLU negative slope. Defaults to 0.2.
        dropout (float, optional): Dropout rate for attention coefficients. Defaults to 0.0.
        use_bias (bool, optional): Whether to add bias terms (both in linear transform and final). Defaults to True.
        kernel_initializer (str, optional): Initializer identifier for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer identifier for bias weights. Defaults to 'zeros'.
        att_initializer (str, optional): Initializer identifier for attention weights. Defaults to 'glorot_uniform'.
        add_self_loops (bool, optional): Whether to add self-loops. Defaults to True.
        **kwargs: Additional arguments passed to the `MessagePassing` base class.
    """
    def __init__(self,
                 output_dim: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 att_initializer: str = 'glorot_uniform',
                 add_self_loops: bool = True,
                 **kwargs):
        # GAT uses sum aggregation
        super().__init__(aggregator='sum', **kwargs)

        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout  # Store as dropout_rate for clarity
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
        self.use_bias = use_bias # Controls bias in *both* places
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer # Used for *both* biases
        self.att_initializer = att_initializer
        self.add_self_loops = add_self_loops

        self.features_per_head = output_dim

        # Placeholder for layers/weights
        self.linear_transform: Optional[layers.Dense] = None
        self.att: Optional[keras.Variable] = None
        self.bias: Optional[keras.Variable] = None # The final bias term

    def build(self, input_shape):
        """ Build the layer weights. """
        input_dim_shape = input_shape[0]
        if not isinstance(input_dim_shape, (list, tuple)) or len(input_dim_shape) != 2:
            raise ValueError(f"Expected features input shape like (N, F), but got {input_dim_shape}")
        node_feature_dim = input_dim_shape[1]
        if node_feature_dim is None:
             raise ValueError("Input feature dimension cannot be None.")

        # Linear transformation for node features (W)
        # --- FIX: Apply bias here *if* self.use_bias is True ---
        self.linear_transform = layers.Dense(
            self.heads * self.features_per_head,
            kernel_initializer=self.kernel_initializer,
            use_bias=self.use_bias, # Apply bias here
            bias_initializer=self.bias_initializer, # Use same initializer
            name="linear_transform"
        )
        self.linear_transform.build((None, node_feature_dim))

        # Attention parameters (a)
        self.att = self.add_weight(
            shape=(1, self.heads, self.features_per_head),
            initializer=initializers.get(self.att_initializer),
            name="att",
            trainable=True
        )

        # --- FIX: Add separate final bias weight *if* self.use_bias is True ---
        if self.use_bias:
            bias_shape = (self.heads * self.output_dim,) if self.concat else (self.output_dim,)
            # Use a distinct name to avoid potential conflicts if Keras reuses names
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=initializers.get(self.bias_initializer), # Use same initializer
                name="final_bias", # Changed name for clarity
                trainable=True
            )
        else:
            self.bias = None

        super().build(input_shape) # Call base build

    def call(self, inputs: Tuple[Any, Any], training: Optional[bool] = None):
        """ Perform GATv2 convolution. """
        x, edge_index = inputs
        edge_index = ops.cast(edge_index, dtype="int32")

        if self.add_self_loops:
            num_nodes = ops.shape(x)[0]
            edge_index = add_self_loops(edge_index, num_nodes)

        # Pass training flag to propagate
        return self.propagate(x, edge_index, training=training)

    def _compute_attention(self, h_i, h_j, target_idx, num_nodes):
        """ Compute attention coefficients for each edge. """
        g_ij = ops.add(h_i, h_j)
        z_ij = ops.leaky_relu(g_ij, negative_slope=self.negative_slope)
        if self.att is None: raise RuntimeError("Attention weights not built.")
        attn_scores = ops.sum(ops.multiply(z_ij, self.att), axis=-1)
        alpha = self._softmax_by_target(attn_scores, target_idx, num_nodes)
        return ops.expand_dims(alpha, -1)

    def _softmax_by_target(self, scores, target_nodes, num_nodes):
        """ Compute softmax of attention coefficients, grouped by target nodes. """
        target_nodes = ops.cast(target_nodes, dtype='int32')
        max_per_target = ops.segment_max(scores, target_nodes, num_segments=num_nodes)
        current_max_size = ops.shape(max_per_target)[0]
        if self.att is None:
            raise RuntimeError("Attention weights not built. Call layer on data first.")
        attn_scores = ops.sum(ops.multiply(z_ij, self.att), axis=-1)

        max_per_edge = ops.take(max_per_target, target_nodes, axis=0)
        exp_alpha = ops.exp(ops.subtract(scores, max_per_edge))

        sum_per_target = ops.segment_sum(exp_alpha, target_nodes, num_segments=num_nodes)
        current_sum_size = ops.shape(sum_per_target)[0]
        if current_sum_size < num_nodes:
             padding_sum = ops.zeros((num_nodes - current_sum_size, ops.shape(scores)[1]), dtype=scores.dtype)
             sum_per_target = ops.concatenate([sum_per_target, padding_sum], axis=0)

        sum_per_edge = ops.take(sum_per_target, target_nodes, axis=0)
        return ops.divide(exp_alpha, ops.add(sum_per_edge, 1e-10))

    def propagate(self, x, edge_index, **kwargs):
        """ Execute the complete GATv2 message passing flow.

        Args:
            x: Node features tensor of shape [N, F]
            edge_index: Edge indices tensor of shape [2, E]
            **kwargs: Additional arguments, including training flag
        """
        # Extract training flag from kwargs
        training = kwargs.get('training', None)
        N = ops.shape(x)[0]
        E = ops.shape(edge_index)[1]

        if self.linear_transform is None or self.att is None:
            raise RuntimeError("Layer weights not built. Call layer on data first.")

        # Apply linear transformation (W*x + b if use_bias=True): [N, F] -> [N, H * F_out]
        x_transformed = self.linear_transform(x) # Bias is applied here now
        # Reshape: [N, H, F_out]
        x_transformed = ops.reshape(x_transformed, [N, self.heads, self.features_per_head])

        source_idx = ops.cast(edge_index[0], dtype='int32')
        target_idx = ops.cast(edge_index[1], dtype='int32')

        h_j = ops.take(x_transformed, source_idx, axis=0)
        h_i = ops.take(x_transformed, target_idx, axis=0)

        # Compute attention coefficients: [E, H, 1]
        alpha = self._compute_attention(h_i, h_j, target_idx, N)

        # Apply dropout to attention coefficients if dropout rate > 0
        if self.dropout_layer is not None:
             alpha = self.dropout_layer(alpha, training=training)

        # Compute messages (apply attention): [E, H, F_out]
        messages = self.message(h_i, h_j, alpha=alpha)

        # Aggregate messages (using base class 'sum' aggregation)
        messages_flat = ops.reshape(messages, [E, self.heads * self.features_per_head])
        aggregated = super().aggregate(messages_flat, target_idx, num_nodes=N) # [N, H * F_out]

        # Final update (concat/average + final bias)
        output = self.update(aggregated)

        return output

    def message(self, x_i, x_j, **kwargs):
        """ Computes messages with attention weights. """
        alpha = kwargs['alpha'] # [E, H, 1]
        return alpha * x_j      # [E, H, F_out]

    def update(self, aggregated):
        """ Final update step: handles multi-head outputs and applies final bias. """
        N = ops.shape(aggregated)[0]
        if self.concat:
            # aggregated is already [N, heads * features_per_head]
            output = aggregated
        else:
            # Average across heads
            aggregated_reshaped = ops.reshape(aggregated, [N, self.heads, self.features_per_head])
            output = ops.mean(aggregated_reshaped, axis=1) # [N, features_per_head]

        # --- FIX: Add the separate final bias term ---
        if self.use_bias and self.bias is not None:
            output = ops.add(output, self.bias)

        return output

    def get_config(self):
        """ Serializes the layer configuration. """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'heads': self.heads,
            'concat': self.concat,
            'negative_slope': self.negative_slope,
            'dropout': self.dropout_rate,  # Use 'dropout' key for compatibility with tests and PyG
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'att_initializer': self.att_initializer,
            'add_self_loops': self.add_self_loops
        })
        return config

    @classmethod
    def from_config(cls, config):
        """ Creates a layer from its config. """
        config.pop('aggr', None)
        # Make sure we use 'dropout' from config (the key is still 'dropout' for compatibility)
        return cls(**config)
