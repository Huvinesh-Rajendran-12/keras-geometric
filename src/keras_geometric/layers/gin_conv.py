from typing import Any, Optional

import keras
from keras import initializers, layers, ops

from .message_passing import MessagePassing


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) Convolution Layer.

    Implements the Graph Isomorphism Network convolution from the paper:
    "How Powerful are Graph Neural Networks?" (https://arxiv.org/abs/1810.00826)

    The layer performs: h' = MLP((1 + ε) * h + Σ_{j∈N(i)} h_j)

    Args:
        output_dim (int): Dimensionality of the output features.
        mlp_hidden (list[int], optional): List of hidden layer dimensions for the MLP.
            Defaults to empty list (single linear layer).
        aggregator (str, optional): Aggregation method. Defaults to 'sum'.
            Must be one of ['mean', 'max', 'sum'].
        eps_init (float, optional): Initial value for the epsilon parameter. Defaults to 0.0.
        train_eps (bool, optional): Whether epsilon is trainable. Defaults to False.
        use_bias (bool, optional): Whether to use bias in dense layers. Defaults to True.
        dropout (float, optional): Dropout rate for MLP layers. Defaults to 0.0.
        kernel_initializer (str, optional): Initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer for bias weights.
            Defaults to 'zeros'.
        activation (str, optional): Activation function for hidden layers.
            Defaults to 'relu'.
        **kwargs: Additional arguments passed to MessagePassing base class.
    """

    def __init__(
        self,
        output_dim: int,
        mlp_hidden: Optional[list[int]] = None,
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
        self.mlp = None
        self.eps: float | None = None
        self.dropout_layers = []

        # Validate aggregator
        assert (
            self.aggregator in ["mean", "max", "sum"]
        ), f"Invalid aggregator: {self.aggregator}. Must be one of ['mean', 'max', 'sum']"

    def build(self, input_shape: Any) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Expected to be [(N, F), (2, E)] or similar shape information
        """
        # Extract input dimension from the first shape (node features)
        if isinstance(input_shape, list) and len(input_shape) >= 1:
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
                dropout_layer = layers.Dropout(self.dropout_rate)
                mlp_layers.append(dropout_layer)
                self.dropout_layers.append(dropout_layer)

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

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """
        Perform GIN convolution.

        Args:
            inputs: List containing [features, edge_index]
                - features: Node features tensor of shape [N, F]
                - edge_index: Edge indices tensor of shape [2, E]
            training: Boolean flag for training mode (affects dropout)

        Returns:
            Output tensor of shape [N, output_dim]
        """
        # Extract inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, edge_index = inputs
        else:
            raise ValueError(
                f"Expected inputs to be [features, edge_index], got {inputs}"
            )

        # Get graph dimensions
        num_nodes = ops.shape(x)[0]
        num_edges = ops.shape(edge_index)[1]

        # Handle empty graph case
        if num_nodes == 0:
            return ops.zeros((0, self.output_dim), dtype=x.dtype)

        # Ensure edge_index is int32
        edge_index = ops.cast(edge_index, dtype="int32")

        # Aggregate neighbor features
        if num_edges > 0:
            aggr_out = self.propagate(x=x, edge_index=edge_index)
        else:
            # No edges - aggregation returns zeros
            aggr_out = ops.zeros_like(x)

        # GIN update: (1 + eps) * x + aggregation
        if self.train_eps:
            # Use learnable epsilon
            h = (1 + self.eps) * x + aggr_out
        else:
            # Use fixed epsilon
            h = (1 + self.eps_init) * x + aggr_out

        # Apply MLP
        if self.mlp is None:
            raise RuntimeError("MLP not initialized. Call build() first.")

        output = self.mlp(h, training=training)

        return output

    def get_config(self) -> dict[str, Any]:
        """Serialize the layer configuration."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "mlp_hidden": self.mlp_hidden,
                "aggregator": self.aggregator,
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
        """Create a layer from its config."""
        return cls(**config)
