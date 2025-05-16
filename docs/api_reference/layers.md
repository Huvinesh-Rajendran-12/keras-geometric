# Layers

## Message Passing

### MessagePassing

```python
class MessagePassing(layers.Layer)
```

Base class for all graph neural network message passing layers.

**Arguments:**

- **aggregator** (_str_): The aggregation method to use ('mean', 'max', or 'sum'). Default is 'mean'.

**Methods:**

- **message(x_i, x_j)**: Computes messages from source node j to target node i. Default returns x_j.
- **update(aggregated)**: Updates node features based on aggregated messages. Default returns the aggregated messages.
- **aggregate(messages, target_idx, num_nodes)**: Aggregates messages based on target indices using the specified aggregation method.
- **propagate(inputs)**: Propagates messages through the graph. Called in the call() method.

## Graph Convolution Layers

### GCNConv

```python
class GCNConv(MessagePassing)
```

Graph Convolutional Network (GCN) layer implementing:
H' = σ(D⁻⁰⁵ Ã D⁻⁰⁵ X W), where Ã = A + I.

**Arguments:**

- **output_dim** (_int_): Dimension of the output features.
- **use_bias** (_bool_): Whether to use a bias vector. Default is True.
- **kernel_initializer** (_str or Initializer_): Initializer for the kernel weights matrix. Default is 'glorot_uniform'.
- **bias_initializer** (_str or Initializer_): Initializer for the bias vector. Default is 'zeros'.
- **add_self_loops** (_bool_): Whether to add self-loops to the adjacency matrix. Default is True.
- **normalize** (_bool_): Whether to apply symmetric normalization. Default is True.

### GATv2Conv

```python
class GATv2Conv(MessagePassing)
```

Graph Attention Network v2 (GATv2) Convolution Layer as described in 'How Attentive are Graph Attention Networks?' (Brody et al., 2021).

Implements the improved Graph Attention mechanism that uses dynamic attention for better expressiveness.

**Arguments:**

- **output_dim** (_int_): Dimensionality of the output features per head.
- **heads** (_int_): Number of multi-head attentions. Default is 1.
- **concat** (_bool_): Whether to concatenate or average multi-head attentions. Default is True.
- **negative_slope** (_float_): LeakyReLU negative slope. Default is 0.2.
- **dropout** (_float_): Dropout rate for attention coefficients. Default is 0.0.
- **use_bias** (_bool_): Whether to add bias terms. Default is True.
- **kernel_initializer** (_str_): Initializer for kernel weights. Default is 'glorot_uniform'.
- **bias_initializer** (_str_): Initializer for bias weights. Default is 'zeros'.
- **att_initializer** (_str_): Initializer for attention weights. Default is 'glorot_uniform'.
- **add_self_loops** (_bool_): Whether to add self-loops to the graph. Default is True.

### SAGEConv

```python
class SAGEConv(MessagePassing)
```

GraphSAGE Convolution Layer as described in 'Inductive Representation Learning on Large Graphs' (Hamilton et al., 2017).

Implements the GraphSAGE layer with customizable aggregation methods including 'pooling' which applies an MLP before aggregation.

**Arguments:**

- **output_dim** (_int_): Dimensionality of the output features.
- **aggregator** (_str_): Aggregation method ('mean', 'max', 'sum', 'pooling'). Default is 'mean'.
- **normalize** (_bool_): Whether to L2-normalize the output embeddings. Default is False.
- **root_weight** (_bool_): If False, exclude the transformed root node features in the output. Default is True.
- **use_bias** (_bool_): Whether to add a bias term. Default is True.
- **activation** (_str | callable | None_): Activation function for the output. Default is 'relu'.
- **pool_activation** (_str | callable | None_): Activation for the pooling aggregator's MLP. Default is 'relu'.
- **kernel_initializer** (_str_): Initializer for kernel weights. Default is 'glorot_uniform'.
- **bias_initializer** (_str_): Initializer for bias weights. Default is 'zeros'.

### GINConv

```python
class GINConv(MessagePassing)
```

Graph Isomorphism Network (GIN) Convolution Layer as described in 'How Powerful are Graph Neural Networks?' (Xu et al., 2019).

Implements the graph isomorphism layer with MLP as the update function.

**Arguments:**

- **output_dim** (_int_): Dimensionality of the output features.
- **mlp_hidden** (_list[int]_): List of hidden layer dimensions for the MLP.
- **aggregator** (_str_): Aggregation method. Default is 'mean'. Must be one of ['mean', 'max', 'sum'].
- **use_bias** (_bool_): Whether to use bias in dense layers. Default is True.
- **kernel_initializer** (_str_): Initializer for kernel weights. Default is 'glorot_uniform'.
- **bias_initializer** (_str_): Initializer for bias weights. Default is 'zeros'.
- **activation** (_str_): Activation function for hidden layers. Default is 'relu'.
