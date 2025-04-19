import keras
from keras import ops


# Helper function to add self-loops (can be moved to a utils module)
def add_self_loops(edge_index, num_nodes):
    """Adds self-loops to edge_index using keras.ops.."""
    # Ensure edge_index has shape (2, E)
    # If edge_index has shape (*, E), we need to extract the correct rows
    if ops.shape(edge_index)[0] != 2:
        # Extract first two rows which represent source and target nodes
        edge_index = ops.stack([edge_index[0], edge_index[1]], axis=0)

    loop_indices = ops.arange(0, num_nodes, dtype=edge_index.dtype)
    self_loops = ops.stack([loop_indices, loop_indices], axis=0)
    edge_index_with_loops = ops.concatenate([edge_index, self_loops], axis=1)
    return edge_index_with_loops

# Helper function to compute GCN normalization (can be moved to a utils module)
def compute_gcn_normalization(edge_index, num_nodes):
    """Computes D^{-1/2} * D^{-1/2} edge weights for GCN normalization using keras.ops."""
    source, target = edge_index[0], edge_index[1]
    ones = ops.ones_like(source, dtype=keras.backend.floatx())
    degrees = ops.segment_sum(
        data=ones,
        segment_ids=target,
        num_segments=num_nodes
    )
    degree_inv_sqrt = ops.power(ops.add(degrees, 1e-12), -0.5)
    degree_inv_sqrt = ops.where(ops.isinf(degree_inv_sqrt), ops.zeros_like(degree_inv_sqrt), degree_inv_sqrt)
    edge_normalization = ops.multiply(ops.take(degree_inv_sqrt, target, axis=0), ops.take(degree_inv_sqrt, source, axis=0))
    return edge_normalization
