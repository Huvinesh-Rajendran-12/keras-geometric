import keras
import numpy as np

from keras_geometric import GATv2Conv

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

# --- 2. Define the GNN Model ---
# Input layers
node_input = keras.Input(shape=(node_features.shape[1],), name="node_features")
edge_input = keras.Input(shape=(2, None), dtype="int32", name="edge_index")

# Apply GATv2 layer with 2 attention heads
gatv2_layer = GATv2Conv(
    output_dim=16,
    heads=2,
    concat=True,
    dropout_rate=0.2,
    negative_slope=0.2
)

# The GATv2Conv layer expects inputs as a list: [node_features, edge_index]
node_embeddings = gatv2_layer([node_input, edge_input])

# Apply a second GATv2 layer that averages the attention heads
output_layer = GATv2Conv(
    output_dim=4,
    heads=2,
    concat=False,  # Average attention heads instead of concatenating
    dropout_rate=0.2
)([node_embeddings, edge_input])

# Create the Keras model
model = keras.Model(inputs=[node_input, edge_input], outputs=output_layer)

# Compile the model (necessary for training)
model.compile(
    optimizer="adam",
    loss=keras.losses.MeanSquaredError()
)

print("\nModel Summary:")
model.summary()

# --- 3. Use the Model (Example: Get embeddings) ---
# Get the node embeddings
output_embeddings = model.predict([node_features, edge_index])

print("\nInput Node Features Shape:", node_features.shape)
print("Edge Index Shape:", edge_index.shape)
print("Output Node Embeddings Shape:", output_embeddings.shape)
print("Output Embeddings:", output_embeddings)

# --- 4. Visualize the Attention Weights (Not directly accessible in Keras) ---
print("\nNote: To visualize attention weights, you would need to modify the GATv2Conv layer")
print("to return attention coefficients as part of the output or as a separate output.")
