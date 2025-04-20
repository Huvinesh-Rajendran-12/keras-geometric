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

### GINConv

```python
class GINConv(MessagePassing)
```

Graph Isomorphism Network (GIN) Convolution Layer.

**Arguments:**

- **output_dim** (_int_): Dimensionality of the output features.
- **mlp_hidden** (_list[int]_): List of hidden layer dimensions for the MLP.
- **aggregator** (_str_): Aggregation method. Default is 'mean'. Must be one of ['mean', 'max', 'sum'].
- **use_bias** (_bool_): Whether to use bias in dense layers. Default is True.
- **kernel_initializer** (_str_): Initializer for kernel weights. Default is 'glorot_uniform'.
- **bias_initializer** (_str_): Initializer for bias weights. Default is 'zeros'.
- **activation** (_str_): Activation function for hidden layers. Default is 'relu'.
