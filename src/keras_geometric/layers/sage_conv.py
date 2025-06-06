from typing import Any, Optional, Union

import keras
from keras import activations, constraints, initializers, layers, ops, regularizers

from keras_geometric.layers.aggregators import AggregatorFactory
from keras_geometric.layers.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    """
    GraphSAGE (SAmple and aggreGatE) Convolution Layer.

    Implements the GraphSAGE operator from the paper:
    "Inductive Representation Learning on Large Graphs" by Hamilton et al. (2017)

    This layer performs neighborhood aggregation and feature transformation following
    the GraphSAGE algorithm, with support for multiple aggregation methods.

    Update Rules:
    - If root_weight=True:  h_v' = σ(W_l·h_v + W_r·AGG(h_u for u in N(v)) + b)
    - If root_weight=False: h_v' = σ(W_r·AGG(h_u for u in N(v)) + b)

    where:
    - h_v is the feature vector of node v
    - N(v) is the neighborhood of node v
    - AGG is the aggregation function (mean, max, sum, or pooling)
    - W_l, W_r are learnable weight matrices
    - b is the bias vector
    - σ is the activation function

    Args:
        output_dim: Dimensionality of the output features.
        aggregator: Aggregation method. One of 'mean', 'max', 'sum', or 'pooling'.
            - 'mean': Average of neighbor features
            - 'max': Element-wise maximum of neighbor features
            - 'sum': Sum of neighbor features
            - 'pooling': Apply MLP before max-pooling
            Defaults to 'mean'.
        normalize: Whether to L2-normalize the output embeddings. Defaults to False.
        root_weight: If True, include transformed root node features in the output.
            If False, only use aggregated neighbor features. Defaults to True.
        use_bias: Whether to add bias terms. Defaults to True.
        activation: Activation function for the final output. Can be a string
            identifier or keras.activations function. Defaults to 'relu'.
        pool_activation: Activation function for the pooling MLP (only used when
            aggregator='pooling'). Defaults to 'relu'.
        pool_hidden_dim: Hidden dimension for the pooling MLP. If None, uses
            the input dimension. Only used when aggregator='pooling'.
        kernel_initializer: Initializer for weight matrices. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Regularizer for weight matrices. Defaults to None.
        bias_regularizer: Regularizer for bias vectors. Defaults to None.
        kernel_constraint: Constraint for weight matrices. Defaults to None.
        bias_constraint: Constraint for bias vectors. Defaults to None.
        dropout_rate: Dropout rate to apply to features. Defaults to 0.0.
        **kwargs: Additional arguments passed to the MessagePassing base class.

    Example:
        ```python
        # Basic usage
        sage = SAGEConv(output_dim=64, aggregator='mean')

        # With pooling aggregator
        sage = SAGEConv(
            output_dim=64,
            aggregator='pooling',
            pool_hidden_dim=32,
            pool_activation='relu'
        )

        # Apply to graph data
        node_features = keras.ops.ones((10, 32))  # 10 nodes, 32 features
        edge_index = keras.ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype='int32')
        output = sage([node_features, edge_index])
        ```
    """

    def __init__(
        self,
        output_dim: int,
        aggregator: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        use_bias: bool = True,
        activation: Optional[str] = "relu",
        pool_activation: Optional[str] = "relu",
        pool_hidden_dim: Optional[int] = None,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Validate aggregator (now includes pooling)
        valid_aggregators = ["mean", "max", "sum", "min", "std", "pooling"]
        if aggregator not in valid_aggregators:
            raise ValueError(
                f"Invalid aggregator '{aggregator}'. Must be one of {valid_aggregators}"
            )

        # For pooling aggregator, we'll set up the pooling MLP in build() and
        # handle it specially. For others, pass directly to base class.
        if aggregator == "pooling":
            # Use mean aggregator in base class, we'll handle pooling manually
            super().__init__(aggregator="mean", **kwargs)
        else:
            super().__init__(aggregator=aggregator, **kwargs)

        # Store the actual aggregator name
        self.actual_aggregator = aggregator

        # Store configuration
        self.output_dim = output_dim
        self.normalize = normalize
        self.root_weight = root_weight
        self.use_bias = use_bias
        self.pool_hidden_dim = pool_hidden_dim
        self.dropout_rate = dropout_rate

        # Initialize activation functions
        self.activation = activations.get(activation)
        self.pool_activation = activations.get(pool_activation)

        # Initialize initializers, regularizers, and constraints
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Layers to be built
        self.lin_neigh = None  # Transform aggregated neighbors
        self.lin_self = None  # Transform self features (if root_weight=True)
        self.pool_mlp = None  # MLP for pooling aggregator
        self.bias = None  # Final bias term

    def build(self, input_shape: Union[list, tuple]) -> None:
        """Build layer weights based on input shapes.

        Args:
            input_shape: Expected to be [(N, F), (2, E)] where N is the number
                of nodes, F is the feature dimension, and E is the number of edges.

        Raises:
            ValueError: If input shape is invalid.
        """
        if input_shape is None:
            return

        # Validate input shape
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                f"Expected input_shape to be [(N, F), (2, E)], got {input_shape}"
            )

        # Extract feature dimension
        node_shape = input_shape[0]
        if hasattr(node_shape, "as_list"):
            shape_list = node_shape.as_list()
            if shape_list is None or len(shape_list) < 2:
                raise ValueError(f"Expected node shape (N, F), got {node_shape}")
            input_dim = shape_list[-1]
        elif isinstance(node_shape, (list, tuple)):
            if len(node_shape) < 2:
                raise ValueError(f"Expected node shape (N, F), got {node_shape}")
            input_dim = node_shape[-1]
        else:
            try:
                input_dim = int(node_shape[-1])
            except (TypeError, IndexError) as e:
                raise ValueError(
                    f"Could not extract input dimension from {node_shape}"
                ) from e

        if input_dim is None or input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")

        # Build pooling MLP if needed
        if self.actual_aggregator == "pooling":
            pool_dim = self.pool_hidden_dim or input_dim
            self.pool_mlp = layers.Dense(
                units=pool_dim,
                activation=self.pool_activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name="pool_mlp",
                dtype=self.dtype,
            )

        # Build neighbor transformation layer
        self.lin_neigh = layers.Dense(
            units=self.output_dim,
            use_bias=False,  # Bias added separately at the end
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            name="linear_neigh",
            dtype=self.dtype,
        )

        # Build self transformation layer if needed
        if self.root_weight:
            self.lin_self = layers.Dense(
                units=self.output_dim,
                use_bias=False,  # Bias added separately at the end
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                name="linear_self",
                dtype=self.dtype,
            )

        # Add final bias if needed
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
                trainable=True,
                dtype=self.dtype,
            )

        # Call parent build
        super().build(input_shape)

    def compute_output_shape(self, input_shape: Union[list, tuple]) -> tuple:
        """Compute the output shape of the layer.

        Args:
            input_shape: Expected to be [(N, F), (2, E)]

        Returns:
            Output shape tuple (N, output_dim)
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            node_shape = input_shape[0]
            if hasattr(node_shape, "as_list"):
                shape_list = node_shape.as_list()
                batch_size = shape_list[0] if shape_list else None
            else:
                batch_size = (
                    node_shape[0] if isinstance(node_shape, (list, tuple)) else None
                )
            return (batch_size, self.output_dim)
        return (None, self.output_dim)

    def message(
        self,
        x_i: keras.KerasTensor,
        x_j: keras.KerasTensor,
        edge_attr: Optional[keras.KerasTensor] = None,
        edge_index: Optional[keras.KerasTensor] = None,
        size: Optional[tuple[int, int]] = None,
        **kwargs: Any,
    ) -> keras.KerasTensor:
        """Compute messages for SAGEConv.
        Args:
            x_i: Target node features [E, F]
            x_j: Source node features [E, F]
            edge_attr: Optional edge attributes (not used in SAGEConv)
            edge_index: Optional edge indices tensor (not used in this method)
            size: Optional graph size tuple (not used in this method)
            **kwargs: Additional arguments including 'training'
        Returns:
            Messages [E, F] or [E, pool_dim] for pooling
        """
        training = kwargs.get("training", None)
        if self.aggregator == "pooling":
            # Apply dropout if training
            if self.dropout_rate > 0 and training:
                dropout_layer = layers.Dropout(self.dropout_rate)
                x_j = dropout_layer(x_j, training=training)
            # Apply pooling MLP before aggregation
            # Add a check to potentially resolve diagnostic 'Expected a callable, got None'
            if self.pool_mlp is None:
                raise RuntimeError(
                    "Pooling MLP not initialized when aggregator is 'pooling'. "
                    "This indicates a build issue or unexpected state."
                )
            return self.pool_mlp(x_j)
        else:
            # Apply dropout if training
            if self.dropout_rate > 0 and training:
                dropout_layer = layers.Dropout(self.dropout_rate)
                x_j = dropout_layer(x_j, training=training)
            return x_j

    def aggregate_neighbors(
        self,
        x: keras.KerasTensor,
        edge_index: keras.KerasTensor,
        num_nodes: int,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Aggregate neighbor features using MessagePassing base class.

        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            num_nodes: Number of nodes

        Returns:
            Aggregated features [N, F] or [N, pool_dim] for pooling
        """
        # Handle empty edge case
        if ops.shape(edge_index)[1] == 0:
            if self.actual_aggregator == "pooling" and self.pool_mlp is not None:
                # For pooling, return zeros with correct output dimension
                dummy_input = ops.zeros((1, ops.shape(x)[1]), dtype=x.dtype)
                dummy_output = self.pool_mlp(dummy_input)
                feature_dim = ops.shape(dummy_output)[1]
            else:
                feature_dim = ops.shape(x)[1]
            return ops.zeros((num_nodes, feature_dim), dtype=x.dtype)

        # Use message passing paradigm properly
        source_node_idx = edge_index[0]
        target_node_idx = edge_index[1]
        x_j = ops.take(x, source_node_idx, axis=0)
        x_i = ops.take(x, target_node_idx, axis=0)
        messages = self.message(x_i, x_j, training=training)

        # Handle pooling aggregator specially
        if self.actual_aggregator == "pooling":
            # Create pooling aggregator and use it
            pooling_aggregator = AggregatorFactory.create_pooling(self.pool_mlp)
            aggregated = pooling_aggregator.aggregate(
                messages, target_node_idx, num_nodes
            )
        else:
            # Use base class aggregation for other aggregators
            aggregated = super().aggregate(
                messages, target_node_idx, num_nodes=num_nodes
            )

        return self.update(aggregated)

    # pyrefly: ignore #bad-override
    def call(
        self,
        inputs: Union[list[keras.KerasTensor], tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward pass of the SAGEConv layer.

        Args:
            inputs: List or tuple containing [node_features, edge_index]
                - node_features: Tensor of shape [N, F]
                - edge_index: Tensor of shape [2, E] or [E, 2]
            training: Boolean flag indicating training mode
            mask: Optional mask tensor (not used in standard SAGEConv)

        Returns:
            Updated node features of shape [N, output_dim]

        Raises:
            ValueError: If inputs format is incorrect
        """
        # Validate inputs
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            raise ValueError(
                "SAGEConv expects inputs to be a list/tuple of [node_features, edge_index]"
            )

        x, edge_index = inputs[0], inputs[1]

        # Ensure correct dtypes
        x = ops.cast(x, self.compute_dtype)
        edge_index = ops.cast(edge_index, "int32")

        # Handle edge_index shape
        edge_shape = ops.shape(edge_index)
        if edge_shape[0] != 2:
            # If shape is [E, 2], transpose to [2, E]
            if edge_shape[1] == 2:
                edge_index = ops.transpose(edge_index)
            else:
                raise ValueError(
                    f"edge_index must have shape [2, E] or [E, 2], but got {edge_shape}"
                )

        # Ensure the layer is built with the correct input shape
        # This handles cases where the layer might be called before build is automatically triggered
        # or if the shape wasn't available during the initial build call.
        if not self.built:
            self.build([ops.shape(x), ops.shape(edge_index)])

        # Get number of nodes
        num_nodes = ops.shape(x)[0]

        # 1. Aggregate neighbor features
        aggregated = self.aggregate_neighbors(
            x, edge_index, num_nodes, training=training
        )

        # 2. Transform aggregated neighbors
        # pyrefly: ignore # implicitly-defined-attribute
        h_neigh = self.lin_neigh(aggregated)

        # 3. Transform and combine self features (if root_weight=True)
        if self.root_weight and self.lin_self is not None:
            # Apply dropout to self features if training
            x_self = x
            # Using the Functional Dropout API
            if self.dropout_rate > 0 and training:
                dropout_layer = layers.Dropout(self.dropout_rate)
                x_self = dropout_layer(x_self)

            h_self = self.lin_self(x_self)
            out = ops.add(h_self, h_neigh)
        else:
            out = h_neigh

        # 4. Add bias
        if self.use_bias and self.bias is not None:
            out = ops.add(out, self.bias)

        # 5. Apply activation
        if self.activation is not None:
            out = self.activation(out)

        # 6. Optional L2 normalization
        if self.normalize:
            out = ops.normalize(out, axis=-1, order=2)

        return out

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "aggregator": self.aggregator,
                "normalize": self.normalize,
                "root_weight": self.root_weight,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "pool_activation": activations.serialize(self.pool_activation),
                "pool_hidden_dim": self.pool_hidden_dim,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "dropout_rate": self.dropout_rate,
            }
        )
        # Remove base aggregator as we store the actual aggregator
        config.pop("aggregator", None)
        config["aggregator"] = self.actual_aggregator
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SAGEConv":
        """Create layer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            New SAGEConv instance
        """
        # Make a copy to avoid modifying the original
        config = config.copy()

        # Deserialize objects
        config["activation"] = activations.deserialize(config.get("activation"))
        config["pool_activation"] = activations.deserialize(
            config.get("pool_activation")
        )
        config["kernel_initializer"] = initializers.deserialize(
            config.get("kernel_initializer", "glorot_uniform")
        )
        config["bias_initializer"] = initializers.deserialize(
            config.get("bias_initializer", "zeros")
        )
        config["kernel_regularizer"] = regularizers.deserialize(
            config.get("kernel_regularizer")
        )
        config["bias_regularizer"] = regularizers.deserialize(
            config.get("bias_regularizer")
        )
        config["kernel_constraint"] = constraints.deserialize(
            config.get("kernel_constraint")
        )
        config["bias_constraint"] = constraints.deserialize(
            config.get("bias_constraint")
        )

        return cls(**config)
