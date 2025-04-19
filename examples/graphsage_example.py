import keras
import numpy as np

from keras_geometric import SAGEConv

"""
This example demonstrates how to use the GraphSAGE convolution layer
for node representation learning in a simple graph.
"""

# --- 1. Prepare Graph Data ---
# Example: A simple graph with 6 nodes and 7 edges
# Node features: 4 features per node
node_features = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0]
], dtype=np.float32)

# Edge index (COO format: [senders, receivers])
# Bidirectional edges: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0, 1-3
edge_index = np.array([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 3],  # Senders
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 3, 1]   # Receivers
], dtype=np.int32)

num_nodes = node_features.shape[0]
input_dim = node_features.shape[1]
hidden_dim = 8
output_dim = 2

# --- 2. Define a GraphSAGE Model ---
# Input layers
node_input = keras.Input(shape=(input_dim,), name="node_features")
edge_input = keras.Input(shape=(2, None), dtype="int32", name="edge_index")

# First GraphSAGE layer (mean aggregation)
x = SAGEConv(
    output_dim=hidden_dim,
    aggregator='mean',
    activation='relu',
    normalize=True
)([node_input, edge_input])

# Second GraphSAGE layer (max aggregation)
output = SAGEConv(
    output_dim=output_dim,
    aggregator='max',
    activation=None,
    normalize=True
)([x, edge_input])

# Create the model
model = keras.Model(inputs=[node_input, edge_input], outputs=output)

print("\n--- Model Summary ---")
model.summary()

# --- 3. Forward Pass ---
print("\n--- Forward Pass ---")
# Get node embeddings
node_embeddings = model.predict([node_features, edge_index])
print("Node embeddings shape:", node_embeddings.shape)
print("Node embeddings (normalized to unit vectors):")
print(node_embeddings)

# --- 4. Visualization for 2D Embeddings ---
print("\n--- Visualization Note ---")
print("Since the output dimension is 2, these embeddings can be directly visualized")
print("in a 2D scatter plot, where similar nodes should be placed closer together.")
print("The GraphSAGE algorithm learns node representations that preserve graph topology")
print("by aggregating information from a node's neighborhood.")

# --- 5. Explain How GraphSAGE Works ---
print("\n--- GraphSAGE Algorithm Summary ---")
print("1. For each node, aggregate features from its neighbors using the specified")
print("   aggregation function (mean, max, or sum).")
print("2. Transform both the node's own features and the aggregated neighborhood")
print("   features using separate weight matrices.")
print("3. Combine the transformed features (usually by addition).")
print("4. Apply non-linearity (e.g., ReLU) and optional normalization.")
print("5. This process can be stacked in multiple layers to capture multi-hop")
print("   neighborhood information.")
