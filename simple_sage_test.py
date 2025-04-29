import keras
import numpy as np

# Import MessagePassing and SAGEConv
from keras_geometric.layers import MessagePassing
from keras_geometric.layers.sage_conv import (
    SAGEConv,  # Direct import since it might not be in __init__
)

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

# Create a SAGE layer
sage_layer = SAGEConv(
    output_dim=output_dim,
    aggregator='mean',
    normalize=False,
    root_weight=True,
    use_bias=True,
    activation=None
)

# Apply the layer directly
output = sage_layer([node_features_tensor, edge_index_tensor])

print("Input shape:", node_features.shape)
print("Edge index shape:", edge_index.shape)
print("Output shape:", output.shape)
print("Output:", keras.ops.convert_to_numpy(output))
