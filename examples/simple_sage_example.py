import numpy as np
import tensorflow as tf

from keras_geometric import SAGEConv

# Set up a simple graph
num_nodes = 6
input_dim = 4
output_dim = 2

# Node features
node_features = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# Edge index (COO format)
edge_index = np.array(
    [
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 3],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 3, 1],  # Target nodes
    ],
    dtype=np.int32,
)

# Convert to tensors
# pyrefly: ignore  # unexpected-keyword
node_features_tensor = tf.convert_to_tensor(node_features, dtype=tf.float32)
# pyrefly: ignore  # unexpected-keyword
edge_index_tensor = tf.convert_to_tensor(edge_index, dtype=tf.int32)

# Create a GraphSAGE layer
sage_layer = SAGEConv(
    output_dim=output_dim, aggregator="mean", normalize=True, activation="relu"
)

# Apply the layer directly
output = sage_layer([node_features_tensor, edge_index_tensor])

print("Input shape:", node_features.shape)
print("Edge index shape:", edge_index.shape)
print("Output shape:", output.shape)
print("Output:", output.numpy())
