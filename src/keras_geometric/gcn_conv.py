import keras
from keras import layers, initializers, ops
# import tensorflow as tf # No longer needed

# Assuming MessagePassing is in the same directory or package
from .message_passing import MessagePassing

# Helper function to add self-loops (can be moved to a utils module)
def add_self_loops(edge_index, num_nodes):
    """Adds self-loops to edge_index using keras.ops."""
    loop_indices = ops.arange(0, num_nodes, dtype=edge_index.dtype)
    self_loops = ops.stack([loop_indices, loop_indices], axis=0)
    edge_index_with_loops = ops.concatenate([edge_index, self_loops], axis=1)
    return edge_index_with_loops

# Helper function to compute GCN normalization (can be moved to a utils module)
def compute_gcn_normalization(edge_index, num_nodes):
    """Computes D^{-1/2} * D^{-1/2} edge weights for GCN normalization using keras.ops."""
    source, target = edge_index[0], edge_index[1]
    ones = ops.ones_like(source, dtype=keras.backend.floatx())
    degrees = ops.segment_sum(
        data=ones,
        segment_ids=target,
        num_segments=num_nodes
    )
    degree_inv_sqrt = ops.power(degrees + 1e-12, -0.5)
    degree_inv_sqrt = ops.where(ops.isinf(degree_inv_sqrt), ops.zeros_like(degree_inv_sqrt), degree_inv_sqrt)
    edge_normalization = ops.take(degree_inv_sqrt, target, axis=0) * ops.take(degree_inv_sqrt, source, axis=0)
    return edge_normalization


class GCNConv(MessagePassing): # Inherit from MessagePassing
    """
    Graph Convolutional Network (GCN) layer implementing:
    H' = σ(D⁻⁰⁵ Ã D⁻⁰⁵ X W)
    where Ã = A + I.

    Inherits from MessagePassing. Normalization and self-loops handled internally.
    Activation is typically applied *after* this layer. Backend Agnostic.

    Args:
        output_dim (int): Dimension of the output features.
        use_bias (bool, optional): Whether to use a bias vector. Defaults to True.
        kernel_initializer (str or Initializer, optional): Initializer for the kernel
            weights matrix (W). Defaults to 'glorot_uniform'.
        bias_initializer (str or Initializer, optional): Initializer for the bias
            vector. Defaults to 'zeros'.
        add_self_loops (bool, optional): Whether to add self-loops to the adjacency
            matrix (Ã = A + I). Defaults to True.
        normalize (bool, optional): Whether to apply symmetric normalization
            (D⁻⁰⁵ Ã D⁻⁰⁵). Defaults to True.
        **kwargs: Additional keyword arguments passed to the base MessagePassing layer.
                 Note: 'aggr' will be overridden to 'add'.
    """
    def __init__(self,
                 output_dim: int,
                 use_bias: bool = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):
        # --- FIX: Remove explicit aggr='add' here. Let kwargs pass it from config. ---
        # Ensure 'aggr' from kwargs doesn't conflict if passed manually AND via config
        # Best practice: Let config drive deserialization.
        # GCN *requires* 'add', so we set it after super() if needed,
        # but rely on base class __init__ to handle 'aggr' from kwargs first.
        # The base class default or value from config should be used.
        # We then force self.aggr = 'add' if necessary, although GCN logic
        # now relies on the overridden aggregate method which uses sum anyway.
        # Let's ensure the base class is initialized correctly via kwargs.
        kwargs['aggr'] = 'sum' # Ensure 'add' is passed if not in kwargs from config
        super().__init__(**kwargs)

        # Override aggr setting if it came from config incorrectly
        # GCN implementation relies on summation in the overridden aggregate method
        self.aggr = 'sum'

        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Pop custom args if they were passed in kwargs, base layer doesn't expect them
        # (This might not be needed if they aren't passed via kwargs anyway)
        # kwargs.pop('add_self_loops', None)
        # kwargs.pop('normalize', None)


    def build(self, input_shape):
        """Build the layer weights (kernel W and bias b)."""
        feat_shape = input_shape[0]
        if not isinstance(feat_shape, (list, tuple)) or len(feat_shape) != 2:
             raise ValueError(f"Expected features input shape like (N, F), but got {feat_shape}")
        input_dim = feat_shape[-1] # F

        self.kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            name='kernel', trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name='bias', trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def aggregate(self, messages, target_idx, num_nodes):
        """
        Overrides base aggregate to use keras.ops.segment_sum with num_segments.
        GCN uses 'add' aggregation.
        """
        aggregated = ops.segment_sum(
            data=messages,
            segment_ids=target_idx,
            num_segments=num_nodes
        )
        return aggregated

    def update(self, aggregated_features):
        """Overrides base update to add bias."""
        if self.use_bias:
            return ops.add(aggregated_features, self.bias)
        else:
            return aggregated_features

    def propagate(self, x, edge_index, edge_weight):
        """Custom propagation for GCN using MessagePassing structure."""
        num_nodes = ops.shape(x)[0]
        x_transformed = ops.matmul(x, self.kernel)
        source_node_indices = edge_index[0]
        target_node_indices = edge_index[1]
        source_features = ops.take(x_transformed, source_node_indices, axis=0)
        messages = source_features * edge_weight[:, None]
        aggregated = self.aggregate(messages, target_node_indices, num_nodes=num_nodes)
        updated = self.update(aggregated)
        return updated

    def call(self, inputs):
        """Performs the GCN convolution using the MessagePassing structure."""
        x, edge_index = inputs
        num_nodes = ops.shape(x)[0]

        if self.add_self_loops:
            edge_index_effective = add_self_loops(edge_index, num_nodes)
        else:
            edge_index_effective = edge_index

        if self.normalize:
            edge_weight = compute_gcn_normalization(edge_index_effective, num_nodes)
        else:
            edge_weight = ops.ones((ops.shape(edge_index_effective)[1],), dtype=x.dtype)

        output = self.propagate(x=x, edge_index=edge_index_effective, edge_weight=edge_weight)
        return output

    def get_config(self):
        """Serializes the layer configuration."""
        config = super().get_config() # Base config includes 'aggr'
        # Ensure aggr is 'add' if base class didn't set it correctly (shouldn't happen)
        config['aggr'] = 'sum'
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'add_self_loops': self.add_self_loops,
            'normalize': self.normalize
        })
        # --- FIX: Remove redundant check ---
        # if 'aggr' not in config:
        #      config['aggr'] = self.aggr # Should be 'add'
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config."""
        # Deserialize initializers before passing to __init__
        config['kernel_initializer'] = initializers.deserialize(config['kernel_initializer'])
        config['bias_initializer'] = initializers.deserialize(config['bias_initializer'])
        # Ensure 'aggr' from config doesn't cause issues if __init__ overrides it.
        # Best practice: __init__ should respect config values when deserializing.
        # The fixed __init__ handles 'aggr' from kwargs correctly now.
        return cls(**config)

