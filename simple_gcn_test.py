import keras
import numpy as np

# Import MessagePassing and GCNConv
from keras_geometric.layers import GCNConv, MessagePassing

# Create a simple graph
num_nodes = 4
input_dim = 3
output_dim = 2

# Node features
node_features = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
], dtype=np.float32)

# Edge index (COO format)
edge_index = np.array([
    [0, 0, 1, 2, 3],  # Source nodes
    [1, 2, 2, 3, 0]   # Target nodes
], dtype=np.int32)

# Convert to tensors
node_features_tensor = keras.ops.convert_to_tensor(node_features)
edge_index_tensor = keras.ops.convert_to_tensor(edge_index)

# Create a GCN layer
gcn_layer = GCNConv(output_dim=output_dim, add_self_loops=True, normalize=True)

# Apply the layer directly
output = gcn_layer([node_features_tensor, edge_index_tensor])

print("Input shape:", node_features.shape)
print("Edge index shape:", edge_index.shape)
print("Output shape:", output.shape)
print("Output:", keras.ops.convert_to_numpy(output))
