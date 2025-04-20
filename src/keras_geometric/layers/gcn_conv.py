# Assuming MessagePassing is in the same directory or package
from keras import ops

from keras_geometric.utils.main import add_self_loops, compute_gcn_normalization

from .message_passing import MessagePassing


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
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):
        # GCN requires sum aggregation, so we pass it directly to the base class
        super().__init__(aggregator='sum', **kwargs)

        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.add_self_loops = add_self_loops
        self.normalize = normalize


    def build(self, input_shape):
        """Build the layer weights (kernel W and bias b).

        Args:
            input_shape: List of input shapes [node_features_shape, edge_index_shape]
                where node_features_shape is (N, F) with N nodes and F features

        Raises:
            ValueError: If the input shape is not as expected
        """
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

    def update(self, aggregated):
        """Overrides base update to add bias.

        Args:
            aggregated: Tensor of shape [N, F] containing the aggregated messages

        Returns:
            Tensor of shape [N, F] containing the updated node features with bias added
        """
        if self.use_bias:
            return ops.add(aggregated, self.bias)
        else:
            return aggregated

    def propagate(self, inputs):
        """Custom propagation for GCN using MessagePassing structure.

        This method implements the GCN-specific message passing logic:
        1. Transform node features using the weight matrix
        2. Compute messages as transformed features weighted by edge weights
        3. Aggregate messages for each target node
        4. Apply bias (in the update step)

        Args:
            inputs: Tuple containing (x, edge_index, edge_weight)
                - x: Tensor of shape [N, F] containing node features
                - edge_index: Tensor of shape [2, E] containing edge indices
                - edge_weight: Tensor of shape [E] containing normalized edge weights

        Returns:
            Tensor of shape [N, output_dim] containing the updated node features
        """
        x, edge_index, edge_weight = inputs
        N = ops.shape(x)[0]
        x_transformed = ops.matmul(x, self.kernel)
        source_node_indices = edge_index[0]
        target_node_indices = edge_index[1]
        source_features = ops.take(x_transformed, source_node_indices, axis=0)
        messages = source_features * edge_weight[:, None]
        aggregated = self.aggregate(messages, target_node_indices, num_nodes=N)
        updated = self.update(aggregated)
        return updated

    def call(self, inputs):
        """Performs the GCN convolution using the MessagePassing structure.

        This method handles the preprocessing steps for GCN:
        1. Ensure edge_index has the correct shape
        2. Add self-loops if specified
        3. Compute normalization coefficients if specified
        4. Call propagate with the prepared inputs

        Args:
            inputs: List containing [x, edge_index]
                - x: Tensor of shape [N, F] containing node features
                - edge_index: Tensor of shape [2, E] containing edge indices

        Returns:
            Tensor of shape [N, output_dim] containing the updated node features
        """
        x, edge_index = inputs
        num_nodes = ops.shape(x)[0]
        edge_index = ops.cast(edge_index, dtype='int32')

        # Ensure edge_index has shape (2, E) if it doesn't
        if ops.shape(edge_index)[0] != 2:
            # Extract first two rows which represent source and target nodes
            edge_index = ops.stack([edge_index[0], edge_index[1]], axis=0)

        if self.add_self_loops:
            edge_index_effective = add_self_loops(edge_index, num_nodes)
        else:
            edge_index_effective = edge_index

        if self.normalize:
            edge_weight = compute_gcn_normalization(edge_index_effective, num_nodes)
        else:
            edge_weight = ops.ones((ops.shape(edge_index_effective)[1],), dtype=x.dtype)

        output = self.propagate(inputs=(x, edge_index_effective, edge_weight))
        return output

    def get_config(self):
        """Serializes the layer configuration."""
        config = super().get_config() # Base config includes 'aggregator'
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'add_self_loops': self.add_self_loops,
            'normalize': self.normalize
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        Args:
            config: Dictionary containing the layer configuration

        Returns:
            A new GCNConv layer instance
        """
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        # Remove 'aggr' and 'aggregator' since 'aggregator' is passed explicitly in __init__
        if 'aggregator' in config_copy:
            config_copy.pop('aggregator')
        return cls(**config_copy)
