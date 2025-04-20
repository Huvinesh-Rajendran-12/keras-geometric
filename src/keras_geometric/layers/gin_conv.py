
import keras
from keras import layers, ops

from .message_passing import MessagePassing


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) Convolution Layer.

    This layer implements the Graph Isomorphism Network (GIN) convolution operation,
    which allows for powerful graph representation learning. It aggregates node
    features using a specified aggregation method and transforms them through a
    multi-layer perceptron (MLP).

    Args:
        output_dim (int): Dimensionality of the output features.
        mlp_hidden (list[int]): List of hidden layer dimensions for the MLP.
        aggregator (str, optional): Aggregation method. Defaults to 'mean'.
            Must be one of ['mean', 'max', 'sum'].
        use_bias (bool, optional): Whether to use bias in dense layers. Defaults to True.
        kernel_initializer (str, optional): Initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer (str, optional): Initializer for bias weights.
            Defaults to 'zeros'.
        activation (str, optional): Activation function for hidden layers.
            Defaults to 'relu'.

    Inherits from MessagePassing layer for graph convolution operations.
    """
    def __init__(self,
            output_dim: int,
            mlp_hidden: list[int],
            aggregator: str = 'mean',
            use_bias: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            activation: str = 'relu',
            **kwargs):
        super(GINConv, self).__init__(aggregator=aggregator, **kwargs)
        self.output_dim = output_dim
        self.mlp_hidden = mlp_hidden
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation

        assert self.aggregator in ['mean', 'max', 'sum'], f"Invalid aggregator: {self.aggregator}. Must be one of ['mean', 'max', 'sum']"


    def build(self, input_shape):
        """
        Build the layer weights

        Args:
            input_shape: [(N, F), (2, E)]
        """
        input_dim = input_shape[0]
        if len(input_dim) != 2:
            raise ValueError(f"Input shape must be (N, F), got {input_dim}")

        mlp_layers = []
        for h_dim in self.mlp_hidden:
            mlp_layers.append(
                layers.Dense(
                units=h_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=self.use_bias,
                name=f"mlp_hidden_{len(mlp_layers) + 1}",
                ))

        mlp_layers.append(
            layers.Dense(
                units=self.output_dim,
                activation=None,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=self.use_bias,
                name="mlp_output"))
        self.mlp = keras.Sequential(mlp_layers)


    def call(self, inputs):
        """
        Perform GIN convolution

        Args:
            inputs: List[features, edge_idx]
                - features: [N, F]
                - edge_idx: [2, E]
        Returns:
            [N, output_dim]
        """
        x, edge_idx = inputs

        aggr_neigh = self.propagate([x, edge_idx])

        # combine self features and aggregated neighbors
        h = ops.add(x, aggr_neigh)
        x = self.mlp(h)
        return x

    def get_config(self):
        """
        Serializes the layer configuration
        """
        config = super(GINConv, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'mlp_hidden': self.mlp_hidden,
            'use_bias': self.use_bias,
            # serialize initializers and activations
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'activation': self.activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.
        """
        # Make a copy of the config to avoid modifying the original
        config_copy = config.copy()
        # Remove 'aggregator' since it's passed explicitly in __init__
        if 'aggregator' in config_copy:
            config_copy.pop('aggregator')
        return cls(**config_copy)
