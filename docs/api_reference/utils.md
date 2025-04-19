# Utils

## Data Utils

### GraphData

```python
class GraphData
```

A data structure for storing and managing graph data.

**Arguments:**

- **x** (*np.ndarray or keras.ops.Tensor*): Node features matrix with shape [num_nodes, num_node_features]
- **edge_index** (*np.ndarray or keras.ops.Tensor*): Edge index matrix with shape [2, num_edges]
- **edge_attr** (*np.ndarray or keras.ops.Tensor, optional*): Edge feature matrix with shape [num_edges, num_edge_features]
- **y** (*np.ndarray or keras.ops.Tensor, optional*): Node-level or graph-level target with arbitrary shape
- **num_nodes** (*int, optional*): Explicit number of nodes (useful for isolated nodes)
- **kwargs**: Additional data to store

**Properties:**

- **num_nodes**: Get the number of nodes in the graph
- **num_edges**: Get the number of edges in the graph
- **num_node_features**: Get the number of node features
- **num_edge_features**: Get the number of edge features

**Methods:**

- **to_dict()**: Convert the graph data to a dictionary
- **to_inputs()**: Convert the graph data to a list of inputs for use with Keras models

### batch_graphs

```python
batch_graphs(graphs: List[GraphData]) -> GraphData
```

Batch multiple graphs into a single large graph with disjoint components.

**Arguments:**

- **graphs** (*List[GraphData]*): List of GraphData objects to batch

**Returns:**

A single GraphData object representing the batched graph

## Graph Utils

### add_self_loops

```python
add_self_loops(edge_index, num_nodes) -> keras.ops.Tensor
```

Adds self-loops to edge_index using keras.ops.

**Arguments:**

- **edge_index** (*keras.ops.Tensor*): Edge index tensor with shape [2, num_edges]
- **num_nodes** (*int*): Number of nodes in the graph

**Returns:**

Edge index tensor with self-loops added

### compute_gcn_normalization

```python
compute_gcn_normalization(edge_index, num_nodes) -> keras.ops.Tensor
```

Computes D^{-1/2} * D^{-1/2} edge weights for GCN normalization using keras.ops.

**Arguments:**

- **edge_index** (*keras.ops.Tensor*): Edge index tensor with shape [2, num_edges]
- **num_nodes** (*int*): Number of nodes in the graph

**Returns:**

Normalized edge weights tensor with shape [num_edges]
