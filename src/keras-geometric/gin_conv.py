
from keras import layers, initializers
from .message_passing import MessagePassing


class GINConv(MessagePassing):
    def __init__(self, output_dim: int, mlp_hidden: list[int], aggr: str = 'mean', use_bias: bool = True, kernel_initializer: initializers.Initializer = 'glorot_uniform', bias_initializer: initializers.Initializer = 'zeros', **kwargs):
        super(GINConv, self).__init__(**kwargs) 
        self.output_dim = output_dim
        self.mlp_hidden = mlp_hidden
        self.aggr = aggr
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        assert self.aggr in ['mean', 'max', 'add'], f"Invalid aggregation method: {self.aggr}. Must be one of ['mean', 'sum', 'add']"
        