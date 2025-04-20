import keras
import numpy as np

from keras_geometric import GCNConv

"""
This example demonstrates how to use the Graph Convolutional Network (GCN) layer
for node representation learning in a simple graph.
"""

# --- 1. Prepare Graph Data ---
# Example: A simple graph with 4 nodes and 5 edges
# Node features (e.g., 3 features per node)
node_features = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
], dtype=np.float32)

# Edge index (COO format: [senders, receivers])
# Edges: 0->1, 0->2, 1->2, 2->3, 3->0
edge_index = np.array([
    [0, 0, 1, 2, 3],  # Senders
    [1, 2, 2, 3, 0]   # Receivers
], dtype=np.int32)

num_nodes = node_features.shape[0]
input_dim = node_features.shape[1]
hidden_dim = 8
output_dim = 2

# --- 2. Define a GCN Model ---
# Input layers
node_input = keras.layers.Input(shape=(input_dim,), name="node_features")
# Note: edge_index should be a 2D tensor with shape (2, E), not (None, 2, None)
edge_input = keras.layers.Input(shape=(2, None), dtype="int32", name="edge_index")

# First GCN layer
x = GCNConv(
    output_dim=hidden_dim,
    add_self_loops=True,
    normalize=True
)([node_input, edge_input])

# Apply activation
x = keras.layers.Activation('relu')(x)

# Second GCN layer
output = GCNConv(
    output_dim=output_dim,
    add_self_loops=True,
    normalize=True
)([x, edge_input])

# Create the model
model = keras.Model(inputs=[node_input, edge_input], outputs=output)

print("\n--- Model Summary ---")
model.summary()

# --- 3. Forward Pass ---
print("\n--- Forward Pass ---")
# Get node embeddings
node_embeddings = model.predict({
    "node_features": node_features,
    "edge_index": edge_index
})
print("Node embeddings shape:", node_embeddings.shape)
print("Node embeddings:")
print(node_embeddings)

# --- 4. Visualization for 2D Embeddings ---
print("\n--- Visualization Note ---")
print("Since the output dimension is 2, these embeddings can be directly visualized")
print("in a 2D scatter plot, where similar nodes should be placed closer together.")
print("The GCN algorithm learns node representations that preserve graph topology")
print("by aggregating information from a node's neighborhood.")

# --- 5. Explain How GCN Works ---
print("\n--- GCN Algorithm Summary ---")
print("1. For each node, aggregate features from its neighbors (and optionally itself)")
print("   using a normalized sum aggregation.")
print("2. Transform the aggregated features using a weight matrix.")
print("3. Apply non-linearity (e.g., ReLU).")
print("4. This process can be stacked in multiple layers to capture multi-hop")
print("   neighborhood information.")
