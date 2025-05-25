from typing import Any, Optional, Union

import keras
from keras import constraints, initializers, layers, ops, regularizers

from keras_geometric.utils.main import add_self_loops, compute_gcn_normalization

from .message_passing import MessagePassing


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) layer implementing:
    H' = σ(D^(-1/2) Ã D^(-1/2) X W + b)
    where Ã = A + I (adjacency matrix with added self-loops).

    This implementation follows the original GCN paper by Kipf & Welling (2017)
    and is optimized for Keras 3 with backend-agnostic operations.

    Key features:
    - Inherits from MessagePassing for modular graph operations
    - Supports optional self-loops and symmetric normalization
    - Backend-agnostic implementation using keras.ops
    - Proper weight initialization and serialization support
    - Type hints for better code clarity and IDE support

    Args:
        output_dim: Dimension of the output features.
        use_bias: Whether to use a bias vector. Defaults to True.
        kernel_initializer: Initializer for the kernel weights matrix.
            Can be a string identifier or keras.initializers.Initializer instance.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias vector.
            Can be a string identifier or keras.initializers.Initializer instance.
            Defaults to 'zeros'.
        kernel_regularizer: Regularizer for the kernel weights matrix.
            Defaults to None.
        bias_regularizer: Regularizer for the bias vector.
            Defaults to None.
        kernel_constraint: Constraint for the kernel weights matrix.
            Defaults to None.
        bias_constraint: Constraint for the bias vector.
            Defaults to None.
        add_self_loops: Whether to add self-loops to the adjacency matrix.
            Defaults to True.
        normalize: Whether to apply symmetric normalization.
            Defaults to True.
        dropout_rate: Dropout rate to apply to messages. Defaults to 0.0.
        **kwargs: Additional keyword arguments passed to the base MessagePassing layer.

    Example:
        ```python
        # Create a GCN layer
        gcn = GCNConv(output_dim=64)

        # Apply to graph data
        node_features = keras.ops.ones((10, 32))  # 10 nodes, 32 features
        edge_index = keras.ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype='int32')
        output = gcn([node_features, edge_index])
        ```
    """

    def __init__(
        self,
        output_dim: int,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[keras.constraints.Constraint] = None,
        bias_constraint: Optional[keras.constraints.Constraint] = None,
        add_self_loops: bool = True,
        normalize: bool = True,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # GCN always uses sum aggregation
        kwargs["aggregator"] = "sum"
        super().__init__(**kwargs)

        # Store configuration
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.dropout_rate = dropout_rate

        # Initialize initializers, regularizers, and constraints
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Initialize weights (will be created in build)
        self.kernel = None  # pyrefly: ignore  # implicitly-defined-attribute
        self.bias = None  # pyrefly: ignore  # implicitly-defined-attribute

        # Store edge weights for the current forward pass
        self._current_edge_weights = None
        self._current_training = None

    def build(self, input_shape: Union[list, tuple]) -> None:
        """Build the layer weights.

        Args:
            input_shape: Expected to be [(N, F), (2, E)] or [TensorShape(N, F), TensorShape(2, E)]
                where N is the number of nodes, F is the input features dimension,
                and E is the number of edges.

        Raises:
            ValueError: If the input shape is not as expected.
        """
        if input_shape is None:
            return

        # Validate input shape structure
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                f"Expected input_shape to be a list/tuple with at least 2 elements "
                f"[(node_features_shape), (edge_index_shape)], but got {input_shape}"
            )

        # Extract node features shape
        node_features_shape = input_shape[0]

        # Handle different shape representations
        if hasattr(node_features_shape, "as_list"):
            # TensorShape object
            shape_list = node_features_shape.as_list()
            if shape_list is None or len(shape_list) < 2:
                raise ValueError(
                    f"Expected node features shape to be (N, F), but got {node_features_shape}"
                )
            input_dim = shape_list[-1]
        elif isinstance(node_features_shape, (list, tuple)):
            if len(node_features_shape) < 2:
                raise ValueError(
                    f"Expected node features shape to be (N, F), but got {node_features_shape}"
                )
            input_dim = node_features_shape[-1]
        else:
            # Try to extract the last dimension
            try:
                input_dim = int(node_features_shape[-1])
            except (TypeError, IndexError) as e:
                raise ValueError(
                    f"Could not extract input dimension from shape {node_features_shape}"
                ) from e

        # Validate input dimension
        if input_dim is None or input_dim <= 0:
            raise ValueError(
                f"Input dimension must be a positive integer, but got {input_dim}"
            )

        # Create kernel weight matrix
        self.kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name="kernel",
            trainable=True,
            dtype=self.dtype,
        )

        # Create bias vector if needed
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
        else:
            self.bias = None

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
        """
        Compute messages from source nodes to target nodes.
        In GCN, the message is the transformed source node features weighted by edge weights.

        Args:
            x_i: Target node features of shape [E, F]
            x_j: Source node features of shape [E, F]
            edge_attr: Edge weights of shape [E] (used for GCN normalization)
            edge_index: Edge indices tensor (not used in this method)
            size: Graph size tuple (not used in this method)
            **kwargs: Additional keyword arguments

        Returns:
            Messages of shape [E, output_dim]
        """
        # Transform source node features
        x_j_transformed = ops.matmul(
            x_j, self.kernel
        )  # pyrefly: ignore  # implicitly-defined-attribute

        # Apply dropout if in training mode
        if self.dropout_rate > 0 and self._current_training:
            dropout_layer = layers.Dropout(self.dropout_rate)
            x_j_transformed = dropout_layer(
                x_j_transformed, training=self._current_training
            )

        # Weight messages by edge weights (GCN normalization)
        if edge_attr is not None:
            messages = x_j_transformed * ops.expand_dims(edge_attr, axis=1)
        else:
            messages = x_j_transformed

        return messages

    def update(
        self, aggregated: keras.KerasTensor, x: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Update node features after aggregation.
        In GCN, this simply adds the bias term if enabled.

        Args:
            aggregated: Aggregated messages of shape [N, output_dim]
            x: Original node features (not used in GCN)

        Returns:
            Updated node features of shape [N, output_dim]
        """
        if (
            self.use_bias and self.bias is not None
        ):  # pyrefly: ignore  # implicitly-defined-attribute
            return ops.add(
                aggregated, self.bias
            )  # pyrefly: ignore  # implicitly-defined-attribute
        return aggregated

    def call(
        self,
        inputs: Union[list[keras.KerasTensor], tuple[keras.KerasTensor]],
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward pass of the GCN layer.

        Args:
            inputs: List or tuple containing [node_features, edge_index]
                - node_features: Tensor of shape [N, F]
                - edge_index: Tensor of shape [2, E] or [E, 2]
            training: Boolean flag indicating training mode
            mask: Optional mask tensor (not used in standard GCN)

        Returns:
            Updated node features of shape [N, output_dim]

        Raises:
            ValueError: If inputs format is incorrect
        """
        # Validate inputs
        if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
            raise ValueError(
                "GCNConv expects inputs to be a list/tuple of "
                "[node_features, edge_index]"
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

        # Get number of nodes
        num_nodes = ops.shape(x)[0]

        # Handle empty graph case
        if num_nodes == 0:
            return ops.zeros((0, self.output_dim), dtype=x.dtype)

        # Add self-loops if specified
        if self.add_self_loops:
            edge_index = add_self_loops(edge_index, num_nodes)

        # Check if there are any edges
        num_edges = ops.shape(edge_index)[1]
        if num_edges == 0:
            # No edges case - just transform features and apply bias
            x_transformed = ops.matmul(
                x, self.kernel
            )  # pyrefly: ignore  # implicitly-defined-attribute
            if self.dropout_rate > 0 and training:
                dropout_layer = layers.Dropout(self.dropout_rate)
                x_transformed = dropout_layer(x_transformed, training=training)
            if (
                self.use_bias and self.bias is not None
            ):  # pyrefly: ignore  # implicitly-defined-attribute
                x_transformed = ops.add(
                    x_transformed, self.bias
                )  # pyrefly: ignore  # implicitly-defined-attribute
            return x_transformed

        # Compute edge weights (normalization coefficients)
        if self.normalize:
            edge_weight = compute_gcn_normalization(edge_index, num_nodes)
        else:
            edge_weight = ops.ones((num_edges,), dtype=self.compute_dtype)

        # Store edge weights and training mode for use in message function
        self._current_edge_weights = edge_weight
        self._current_training = training

        # Use the base class propagate method with edge weights as edge_attr
        output = super().propagate(
            x=x, edge_index=edge_index, edge_attr=edge_weight, training=training
        )

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "add_self_loops": self.add_self_loops,
                "normalize": self.normalize,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GCNConv":
        """Create layer from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            New GCNConv instance
        """
        # Make a copy to avoid modifying the original
        config = config.copy()

        # Deserialize objects
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

        # Remove 'aggregator' as it's set in __init__
        config.pop("aggregator", None)

        return cls(**config)
