import keras
from keras import layers, initializers

class GCNConv(layers.Layer):
    def __init__(self, output_dim: int, aggr: str = 'mean', use_bias: bool = True, kernel_initializer: initializers.Initializer = 'glorot_uniform', bias_initializer: initializers.Initializer = 'zeros', **kwargs):
        super(GCNConv, self).__init__(**kwargs) 
        self.output_dim = output_dim
        self.aggr = aggr
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        assert self.aggr in ['mean', 'max', 'sum'], f"Invalid aggregation method: {self.aggr}. Must be one of ['mean', 'max', 'sum']"

    def build(self, input_shape):
        """
        Build the layer weights 

        Args:
            input_shape: [(N, F), (N, N)]
        """
        input_dim = input_shape[0]
        feat_dim = input_dim[-1]

        self.self_kernel = self.add_weight(
            shape=(feat_dim, self.output_dim),
            initializer=self.kernel_initializer,
            name='self_kernel',
            trainable=True
        )

        self.neigh_kernel = self.add_weight(
            shape=(feat_dim, self.output_dim),
            initializer=self.kernel_initializer,
            name='neigh_kernel',
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                name='bias',
                trainable=True
            )
        else:
            self.bias = None
        
        self.built = True

    def call(self, inputs):
        """
        Args:
            inputs: List[features, adjacency]
                - features: [N, F]
                - adjacency: [N, N]
        """
        features, adjacency = inputs
        self_features = keras.ops.matmul(features, self.self_kernel)
        neigh_features = keras.ops.matmul(features, self.neigh_kernel)
        neigh_agg = keras.ops.matmul(adjacency, neigh_features)

        if self.aggr == 'mean':
            output = (self_features + neigh_agg) / 2
        elif self.aggr == 'max':
            output = keras.ops.maximum(self_features,neigh_agg)
        elif self.aggr == 'sum':
            output = keras.ops.add(self_features,neigh_agg)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggr}")

        if self.use_bias:
            output = keras.ops.add(output, self.bias)
        return output

    def get_config(self):
        config = super(GCNConv, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'aggr': self.aggr,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer
        })
        return config
    