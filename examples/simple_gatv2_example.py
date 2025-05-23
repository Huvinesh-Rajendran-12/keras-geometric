import numpy as np
import tensorflow as tf

from keras_geometric import GATv2Conv

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

# Create a GATv2 layer
# Note: There's a bug in the GATv2Conv layer where it uses hidden_channels instead of hidden_dim
# Let's use a single head to avoid the issue
gatv2_layer = GATv2Conv(
    output_dim=output_dim,
    heads=1,  # Use single head to avoid the bug
    concat=False,
    dropout_rate=0.0,
)

# Apply the layer directly
output = gatv2_layer([node_features_tensor, edge_index_tensor])

print("Input shape:", node_features.shape)
print("Edge index shape:", edge_index.shape)
print("Output shape:", output.shape)
print("Output:", output.numpy())
