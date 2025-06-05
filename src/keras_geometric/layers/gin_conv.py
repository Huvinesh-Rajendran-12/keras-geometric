from typing import Any, Optional, Union

import keras
from keras import initializers, layers, ops
from keras.src.ops import KerasTensor

from .message_passing import MessagePassing


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) Convolution Layer.

    Implements the Graph Isomorphism Network convolution from the paper:
    "How Powerful are Graph Neural Networks?" (https://arxiv.org/abs/1810.00826)

    The layer performs: h' = MLP((1 + ε) * h + Σ_{j∈N(i)} h_j)

    Args:
        output_dim: Dimensionality of the output features.
        mlp_hidden: List of hidden layer dimensions for the MLP.
            Defaults to empty list (single linear layer).
        aggregator: Aggregation method. Defaults to 'sum'.
            Must be one of ['mean', 'max', 'sum'].
        eps_init: Initial value for the epsilon parameter. Defaults to 0.0.
        train_eps: Whether epsilon is trainable. Defaults to False.
        use_bias: Whether to use bias in dense layers. Defaults to True.
        dropout: Dropout rate for MLP layers. Defaults to 0.0.
        kernel_initializer: Initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias weights.
            Defaults to 'zeros'.
        activation: Activation function for hidden layers.
            Defaults to 'relu'.
        **kwargs: Additional arguments passed to MessagePassing base class.
    """

    def __init__(
        self,
        output_dim: int,
        mlp_hidden: list[int] | None = None,
        aggregator: str = "sum",  # GIN typically uses sum aggregation
        eps_init: float = 0.0,
        train_eps: bool = False,
        use_bias: bool = True,
        dropout: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(aggregator=aggregator, **kwargs)

        # Store configuration
        self.output_dim = output_dim
        self.mlp_hidden = mlp_hidden if mlp_hidden is not None else []
        self.eps_init = eps_init
        self.train_eps = train_eps
        self.use_bias = use_bias
        self.dropout_rate = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation

        # Initialize attributes that will be defined in build
        self.mlp = None  # pyrefly: ignore  # implicitly-defined-attribute
        self.eps: Union[float, keras.Variable, None] = (
            None  # pyrefly: ignore  # implicitly-defined-attribute
        )

        # Cache for edge indices
        self._cached_edge_idx: Optional[KerasTensor] = (
            None  # pyrefly: ignore  # implicitly-defined-attribute
        )
        self._cached_edge_idx_hash: Optional[int] = (
            None  # pyrefly: ignore  # implicitly-defined-attribute
        )

        # Validate aggregator
        if self.aggregator not in ["mean", "max", "sum"]:
            raise ValueError(
                f"Invalid aggregator: {self.aggregator}. "
                f"Must be one of ['mean', 'max', 'sum']"
            )

    def build(self, input_shape: Any) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Expected to be [(N, F), (2, E)] or similar shape information
        """
        # Extract input dimension from the first shape (node features)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            node_feature_shape = input_shape[0]
        else:
            # Handle case where input_shape might be a single shape
            node_feature_shape = input_shape

        # Handle different shape formats
        if isinstance(node_feature_shape, (list, tuple)):
            if len(node_feature_shape) < 2:
                raise ValueError(
                    f"Expected node features shape (N, F), got {node_feature_shape}"
                )
            input_dim = node_feature_shape[1]
        elif hasattr(node_feature_shape, "__len__") and len(node_feature_shape) >= 2:
            input_dim = int(node_feature_shape[1])
        else:
            raise ValueError(
                f"Cannot extract input dimension from {node_feature_shape}"
            )

        if input_dim is None:
            raise ValueError("Input feature dimension cannot be None")

        # Initialize epsilon parameter
        if self.train_eps:
            self.eps = self.add_weight(
                name="eps",
                shape=(1,),
                initializer=initializers.Constant(self.eps_init),
                trainable=True,
            )
        else:
            self.eps = self.eps_init

        # Build MLP
        mlp_layers = []

        # Add hidden layers with activation and dropout
        for i, hidden_dim in enumerate(self.mlp_hidden):
            mlp_layers.append(
                layers.Dense(
                    units=hidden_dim,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    use_bias=self.use_bias,
                    name=f"mlp_hidden_{i}",
                )
            )
            if self.dropout_rate > 0:
                mlp_layers.append(layers.Dropout(self.dropout_rate))

        # Add output layer (no activation on final layer)
        mlp_layers.append(
            layers.Dense(
                units=self.output_dim,
                activation=None,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=self.use_bias,
                name="mlp_output",
            )
        )

        # Create sequential model
        self.mlp = keras.Sequential(mlp_layers, name="gin_mlp")

        # Build the MLP with the correct input shape
        self.mlp.build((None, input_dim))

        super().build(input_shape)

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
        Compute messages from source node j to target node i.

        For GIN, the message is simply the source node features x_j.

        Args:
            x_i: Tensor of shape [E, F] containing features of the target nodes.
            x_j: Tensor of shape [E, F] containing features of the source nodes.
            edge_attr: Optional tensor of shape [E, D] containing edge attributes.
                Ignored in GIN as it doesn't use edge features.
            edge_index: Optional tensor of shape [2, E] containing the edge indices.
            size: Optional tuple (N_i, N_j) indicating the number of target and source nodes.
            **kwargs: Additional arguments.

        Returns:
            Tensor of shape [E, F] containing the messages (source node features).
        """
        # GIN message is simply the source node features
        return x_j

    def update(
        self, aggregated: KerasTensor, x: KerasTensor | None = None
    ) -> KerasTensor:
        """
        Update node features based on aggregated messages.

        Implements the GIN update: h' = MLP((1 + ε) * h + aggregated)

        Args:
            aggregated: Tensor of shape [N, F] containing the aggregated messages.
            x: Tensor of shape [N, F] containing the original node features.

        Returns:
            Tensor of shape [N, output_dim] containing the updated node features.
        """
        if x is None:
            raise ValueError("Original node features x are required for GIN update")

        if self.mlp is None:
            raise RuntimeError("MLP not initialized. Call build() first.")

        # GIN update: (1 + eps) * x + aggregation
        if self.train_eps:
            # Use learnable epsilon
            h = (1 + self.eps) * x + aggregated
        else:
            # Use fixed epsilon
            h = (1 + self.eps_init) * x + aggregated

        # Apply MLP
        return self.mlp(h)

    # pyrefly: ignore #not-callable
    def call(
        self,
        inputs: Union[list[KerasTensor], tuple[KerasTensor, ...]],
        edge_attr: Optional[KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> KerasTensor:
        """
        Forward pass for the GIN layer.

        Args:
            inputs: List containing [x, edge_index].
                - x: Node features tensor of shape [N, F]
                - edge_index: Edge indices tensor of shape [2, E]
            edge_attr: Optional edge attributes (ignored in GIN).
            training: Whether the layer is in training mode.

        Returns:
            Output tensor of shape [N, output_dim].
        """
        # Parse inputs
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "Inputs must be a list or tuple containing [x, edge_index]"
            )

        if len(inputs) < 2:
            raise ValueError("Inputs must contain at least [x, edge_index]")

        x = inputs[0]
        edge_index = inputs[1]

        # Handle empty graph case
        num_nodes = ops.shape(x)[0]
        if num_nodes == 0:
            return ops.zeros((0, self.output_dim), dtype=x.dtype)

        # Handle graph with nodes but no edges.
        # The GIN update for a node i with no neighbors N(i) is:
        # h'_i = MLP((1 + ε) * h_i + Σ_{j∈N(i)} h_j)
        # If N(i) is empty, the sum is 0. So, h'_i = MLP((1 + ε) * h_i).
        # We apply this simplified logic directly to maintain correct output dimension.
        num_edges = ops.shape(edge_index)[1]
        if num_edges == 0:
            if self.train_eps:
                # Use learnable epsilon
                h = (1 + self.eps) * x
            else:
                # Use fixed epsilon
                h = (1 + self.eps_init) * x
            # Apply MLP to get the correct output dimension
            if self.mlp is None:
                raise RuntimeError("MLP not initialized. This indicates a build issue.")
            return self.mlp(h, training=training)  # pyrefly: ignore #not-callable

        # Cast edge_index to int32 and cache if needed
        edge_index_hash = (
            hash(edge_index.ref())
            if hasattr(edge_index, "ref")
            else id(edge_index)  # pyrefly: ignore #missing-attribute
        )
        if (
            self._cached_edge_idx is None
            or self._cached_edge_idx_hash != edge_index_hash
        ):
            self._cached_edge_idx = ops.cast(edge_index, dtype="int32")
            self._cached_edge_idx_hash = edge_index_hash

        edge_index = self._cached_edge_idx

        # Propagate through the message passing framework
        return self.propagate(
            x=x, edge_index=edge_index, edge_attr=edge_attr, training=training
        )

    # pyrefly: ignore #bad-return
    def compute_output_shape(
        self,
        input_shape: Union[list[tuple[int, ...]], tuple[tuple[int, ...], ...]],
    ) -> tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape(s) of input tensors.

        Returns:
            Output shape tuple.
        """
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape[0] if len(input_shape) > 0 else input_shape

        # Output shape: (batch_size, output_dim)
        return (x_shape[0], self.output_dim)

    def get_config(self) -> dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "mlp_hidden": self.mlp_hidden,
                "eps_init": float(self.eps_init),
                "train_eps": self.train_eps,
                "use_bias": self.use_bias,
                "dropout": self.dropout_rate,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GINConv":
        """
        Creates a layer from its configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            New layer instance.
        """
        return cls(**config)
