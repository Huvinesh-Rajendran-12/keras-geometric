import numpy as np
import tensorflow as tf

from keras_geometric import GCNConv

# Set up a simple graph
num_nodes = 4
input_dim = 3
output_dim = 2

# Node features
node_features = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    dtype=np.float32,
)

# Edge index (COO format)
edge_index = np.array(
    [
        [0, 0, 1, 2, 3],  # Source nodes
        [1, 2, 2, 3, 0],  # Target nodes
    ],
    dtype=np.int32,
)

# Convert to tensors
node_features_tensor = tf.convert_to_tensor(node_features, dtype=tf.float32)
edge_index_tensor = tf.convert_to_tensor(edge_index, dtype=tf.int32)

# Create a GCN layer
gcn_layer = GCNConv(output_dim=output_dim, add_self_loops=True, normalize=True)

# Apply the layer directly
output = gcn_layer([node_features_tensor, edge_index_tensor])

print("Input shape:", node_features.shape)
print("Edge index shape:", edge_index.shape)
print("Output shape:", output.shape)
print("Output:", output.numpy())
