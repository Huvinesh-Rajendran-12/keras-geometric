from typing import Any, Optional, Tuple

import keras
from keras import activations, initializers, layers, ops

# Assuming MessagePassing is in the same directory
from .message_passing import MessagePassing


class SAGEConv(MessagePassing):
    """
    SAGE Convolution Layer (PyG Style Update Rule).

    Implements the GraphSAGE operator similar to PyTorch Geometric's SAGEConv,
    but also includes support for the 'pooling' aggregator variant.

    Update Rule (if root_weight=True):
      h_v' = activation( W_l * h_v + W_r * AGG(neighbors) + b )
    Update Rule (if root_weight=False):
      h_v' = activation( W_r * AGG(neighbors) + b )

    Aggregation (`AGG`):
      - 'mean', 'max', 'sum': Direct aggregation of neighbor features.
      - 'pooling': Applies a Dense layer + activation to neighbor features
                   before max pooling.

    Args:
        output_dim (int): Dimensionality of the output features.
        aggregator (str, optional): Aggregation method ('mean', 'max', 'sum', 'pooling').
            Defaults to 'mean'.
        normalize (bool, optional): Whether to L2-normalize the output embeddings.
            Defaults to False.
        root_weight (bool, optional): If `False`, do not include the transformed
            root node features (W_l * h_v) in the final output. Defaults to True.
        use_bias (bool, optional): Whether to add a final bias term and bias in pool_mlp.
            Defaults to True.
        activation (str | callable | None, optional): Activation function identifier
            to apply to the final output. Defaults to 'relu'. Can be None for no activation.
        pool_activation (str | callable | None, optional): Activation function for the
            pooling aggregator's MLP. Defaults to 'relu'.
        kernel_initializer (str, optional): Initializer identifier for the kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer identifier for the bias weights.
            Defaults to 'zeros'.
        **kwargs: Additional arguments passed to the `MessagePassing` base class.
    """
    def __init__(self,
                 output_dim: int,
                 aggregator: str = 'mean',
                 normalize: bool = False,
                 root_weight: bool = True,
                 use_bias: bool = True,
                 activation: Optional[str] = 'relu',
                 pool_activation: Optional[str] = 'relu', # Activation for pooling MLP
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 **kwargs):

        # List of valid aggregators
        valid_aggregators = ['mean', 'max', 'sum', 'pooling']
        if aggregator not in valid_aggregators:
            raise ValueError(f"Invalid aggregator '{aggregator}'. Must be one of {valid_aggregators}")

        # Determine base aggregator needed for MessagePassing propagate()
        # For 'pooling', we handle aggregation manually in propagate()
        base_aggregator = aggregator if aggregator in ['mean', 'max', 'sum'] else 'max' # Use max as base for pooling
        super().__init__(aggregator=base_aggregator, **kwargs)

        # Store the originally requested aggregator
        self.aggregator = aggregator

        self.output_dim = output_dim
        self.normalize = normalize
        self.root_weight = root_weight
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.pool_activation = activations.get(pool_activation) # Get pool activation object
        # Store initializer identifiers/configs
        self._kernel_initializer_config = kernel_initializer
        self._bias_initializer_config = bias_initializer

        # Linear layers for neighbor and self transformations
        self.lin_neigh: Optional[layers.Dense] = None
        self.lin_self: Optional[layers.Dense] = None
        # Separate bias term added at the end
        self.bias: Optional[keras.Variable] = None
        # MLP for pooling aggregator
        self.pool_mlp: Optional[layers.Dense] = None


    def build(self, input_shape: Tuple[Tuple[Optional[int], int], Tuple[int, Optional[int]]]):
        """ Build layer weights based on input shapes. """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be [(N, F), (2, E)], got {input_shape}")
        if len(input_shape[0]) != 2 or len(input_shape[1]) != 2:
             raise ValueError(f"Expected input_shape to be [(N, F), (2, E)], got {input_shape}")

        in_channels = input_shape[0][1]
        if in_channels is None:
             raise ValueError("Input feature dimension (F) cannot be None for build.")

        # --- FIX: Add pool_mlp creation ---
        if self.aggregator == 'pooling':
            self.pool_mlp = layers.Dense(
                in_channels, # Pool MLP maps F_in -> F_in
                activation=self.pool_activation, # Apply activation within MLP
                kernel_initializer=self._kernel_initializer_config,
                bias_initializer=self._bias_initializer_config,
                use_bias=self.use_bias, # Use bias in pool MLP if enabled
                name='pool_mlp'
            )

        # Layer to transform aggregated neighbor features (W_r)
        self.lin_neigh = layers.Dense(
            units=self.output_dim,
            use_bias=False, # Final bias added separately
            kernel_initializer=self._kernel_initializer_config,
            name="linear_neigh"
        )

        # Layer to transform self features (W_l) - only if root_weight is True
        if self.root_weight:
            self.lin_self = layers.Dense(
                units=self.output_dim,
                use_bias=False, # Final bias added separately
                kernel_initializer=self._kernel_initializer_config,
                name="linear_self"
            )
            self.lin_self.build((None, in_channels)) # Input dim is F_in
        else:
            self.lin_self = None

        # Add separate bias weight if use_bias=True
        if self.use_bias:
             self.bias = self.add_weight(
                 shape=(self.output_dim,),
                 initializer=initializers.get(self._bias_initializer_config),
                 name='bias',
                 trainable=True
             )
        else:
             self.bias = None

    # Override propagate to implement the specific SAGE update logic
    def propagate(self, inputs):
        """
        Overrides propagate for PyG-style GraphSAGE logic.
        Handles mean, max, sum, and pooling aggregators.

        Args:
            x (Tensor): Node features [N, F_in].
            edge_index (Tensor): Graph connectivity [2, E].

        Returns:
            Tensor: Output node features [N, output_dim].
        """
        x, edge_index = inputs
        num_nodes = ops.shape(x)[0]
        source_idx, target_idx = edge_index[0], edge_index[1]

        # --- 1. Aggregate neighbor features ---
        aggregated_neighbors: Any
        if self.aggregator in ['mean', 'max', 'sum']:
            # Use base class propagate which calls message -> aggregate -> update(identity)
            aggregated_neighbors = super().propagate(inputs=(x, edge_index)) # Shape: [N, F_in]

        elif self.aggregator == 'pooling':
            if self.pool_mlp is None: raise RuntimeError("Pool MLP not built.")
            neighbor_feats = ops.take(x, source_idx, axis=0) # [E, F_in]
            # Apply MLP to each neighbor feature *before* max pooling
            h_neighbors_mlp = self.pool_mlp(neighbor_feats) # [E, F_in]
            # Aggregate using segment_max (pooling aggregator typically uses max)
            aggr = ops.segment_max(
                data=h_neighbors_mlp,
                segment_ids=target_idx,
                num_segments=num_nodes
            )
            # Replace potential -inf (for nodes with no neighbors) with zeros
            aggregated_neighbors = ops.where(
                ops.isinf(aggr),
                ops.zeros_like(aggr),
                aggr
            ) # Shape: [N, F_in]
        else:
            # Should be unreachable due to __init__ validation
            raise ValueError(f"Internal error: Unsupported aggregator '{self.aggregator}'")
        return aggregated_neighbors

    def call(self, inputs: Tuple[Any, Any]) -> Any:
        """ Forward pass for GraphSAGE. """
        x, edge_index = inputs
        edge_index = ops.cast(edge_index, dtype='int32')
        # Call the overridden propagate method
        inputs = (x, edge_index)
        aggregated = self.propagate(inputs)
        # --- 2. Transform aggregated neighbors (W_r * h_N(v)) ---
        if self.lin_neigh is None: raise RuntimeError("lin_neigh layer not built.")
        h_neigh = self.lin_neigh(aggregated) # Shape: [N, output_dim] (Bias is NOT added here)

        # --- 3. Transform self features and combine (if root_weight=True) ---
        if self.root_weight:
            if self.lin_self is None: raise RuntimeError("lin_self layer not built.")
            h_self = self.lin_self(x) # Shape: [N, output_dim] (Bias is NOT added here)
            out = ops.add(h_self, h_neigh) # W_l*h + W_r*h_agg
        else:
            out = h_neigh # Only use aggregated neighbors

        # --- 4. Add final bias term here (if enabled) ---
        if self.use_bias and self.bias is not None:
            out = ops.add(out, self.bias)

        # --- 5. Apply Activation Function (if specified) ---
        if self.activation is not None:
            out = self.activation(out)

        # --- 6. Optional L2 Normalization ---
        if self.normalize:
            out = ops.normalize(out, axis=-1, order=2)
        return out


    def get_config(self):
        """Serializes the layer configuration."""
        config = super().get_config() # Includes base 'aggregator'
        # Store the specific aggregator used by this layer
        config['aggregator'] = self.aggregator
        config.update({
            'output_dim': self.output_dim,
            'normalize': self.normalize,
            'root_weight': self.root_weight,
            'use_bias': self.use_bias,
            'activation': activations.serialize(self.activation),
            # Save original initializer identifiers/configs
            'kernel_initializer': self._kernel_initializer_config,
            'bias_initializer': self._bias_initializer_config,
            # Add pool_activation if it exists (only relevant if aggregator='pooling')
            'pool_activation': activations.serialize(getattr(self, 'pool_activation', None))
        })
        # Remove project_input if it was added previously
        config.pop('project_input', None)
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its configuration."""
        # Pop pool_activation before passing to init if needed,
        # or handle it in __init__ if activation objects are passed
        # Assuming __init__ takes identifiers, Keras handles deserialization
        return cls(**config)
