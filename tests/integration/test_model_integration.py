"""
Integration tests for multi-layer GNN models.

These tests verify that different GNN layers work correctly when combined
in realistic model architectures.
"""

import os

import keras
import numpy as np
import pytest

# Set backend before importing keras_geometric
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras_geometric.layers import GATv2Conv, GCNConv, GINConv, SAGEConv

pytestmark = pytest.mark.integration


class TestModelIntegration:
    """Test integration of multiple GNN layers in complete models."""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        num_nodes = 50
        num_edges = 200
        input_dim = 16

        # Random node features
        node_features = np.random.randn(num_nodes, input_dim).astype(np.float32)

        # Random edge indices (ensuring valid node indices)
        edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(
            np.int32
        )

        # Optional edge attributes
        edge_attrs = np.random.randn(num_edges, 8).astype(np.float32)

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "edge_attrs": edge_attrs,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "input_dim": input_dim,
        }

    def test_multi_layer_gcn_model(self, sample_graph_data):
        """Test a multi-layer GCN model for node classification."""
        data = sample_graph_data
        hidden_dim = 32
        output_dim = 7  # Number of classes

        # Build model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, None), name="edge_indices")

        # First GCN layer
        x = GCNConv(hidden_dim, use_bias=True)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.2)(x)

        # Second GCN layer
        x = GCNConv(hidden_dim, use_bias=True)([x, edge_input])
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.2)(x)

        # Classification head
        x = GCNConv(output_dim, use_bias=True)([x, edge_input])
        outputs = keras.layers.Activation("softmax")(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        assert predictions.shape == (data["num_nodes"], output_dim)
        assert np.allclose(
            np.sum(predictions.numpy(), axis=1), 1.0, atol=1e-6
        )  # Softmax check

    def test_heterogeneous_layer_model(self, sample_graph_data):
        """Test model combining different types of GNN layers."""
        data = sample_graph_data
        hidden_dim = 24
        output_dim = 16

        # Build heterogeneous model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        # GCN layer
        x1 = GCNConv(hidden_dim)([node_input, edge_input])
        x1 = keras.layers.Activation("relu")(x1)

        # GAT layer
        x2 = GATv2Conv(hidden_dim, heads=2)([node_input, edge_input])
        x2 = keras.layers.Activation("relu")(x2)

        # SAGE layer
        x3 = SAGEConv(hidden_dim, aggregator="mean")([node_input, edge_input])
        x3 = keras.layers.Activation("relu")(x3)

        # Combine features
        x_combined = keras.layers.Concatenate()([x1, x2, x3])

        # Final layer
        outputs = keras.layers.Dense(output_dim, activation="relu")(x_combined)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        # Expected combined dim would be hidden_dim * 3 (GCN + GAT + SAGE)
        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_graph_classification_model(self, sample_graph_data):
        """Test graph-level classification using pooling."""
        data = sample_graph_data
        hidden_dim = 32
        num_classes = 5

        # Build graph classification model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        # Node embedding layers
        x = GINConv(hidden_dim, aggregator="sum")([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.3)(x)

        x = GINConv(hidden_dim, aggregator="sum")([x, edge_input])
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.3)(x)

        # Graph-level pooling (simple mean pooling)
        graph_embedding = keras.layers.GlobalAveragePooling1D()(
            keras.layers.Reshape((-1, 1))(x)
        )
        graph_embedding = keras.layers.Reshape((hidden_dim,))(graph_embedding)

        # Classification head
        outputs = keras.layers.Dense(num_classes, activation="softmax")(graph_embedding)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        assert predictions.shape == (num_classes,)  # Single graph prediction
        assert np.allclose(np.sum(predictions.numpy()), 1.0, atol=1e-6)  # Softmax check

    def test_sage_with_different_aggregators(self, sample_graph_data):
        """Test SAGEConv with different aggregation strategies in one model."""
        data = sample_graph_data
        hidden_dim = 16
        output_dim = 8

        # Build model with different SAGE aggregators
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        # Different aggregation strategies
        x_mean = SAGEConv(hidden_dim, aggregator="mean")([node_input, edge_input])
        x_mean = keras.layers.Activation("relu")(x_mean)

        x_max = SAGEConv(hidden_dim, aggregator="max")([node_input, edge_input])
        x_max = keras.layers.Activation("relu")(x_max)

        x_sum = SAGEConv(hidden_dim, aggregator="sum")([node_input, edge_input])
        x_sum = keras.layers.Activation("relu")(x_sum)

        # Combine different aggregations
        x_combined = keras.layers.Concatenate()([x_mean, x_max, x_sum])

        # Final transformation
        outputs = keras.layers.Dense(output_dim, activation="relu")(x_combined)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_attention_mechanism_model(self, sample_graph_data):
        """Test model with attention mechanisms."""
        data = sample_graph_data
        hidden_dim = 20
        output_dim = 10

        # Build attention-based model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        # Multi-head attention layers
        x = GATv2Conv(hidden_dim, heads=4, use_bias=True)([node_input, edge_input])
        x = keras.layers.Activation("elu")(x)
        x = keras.layers.Dropout(0.1)(x)

        x = GATv2Conv(hidden_dim, heads=2, use_bias=True)([x, edge_input])
        x = keras.layers.Activation("elu")(x)
        x = keras.layers.Dropout(0.1)(x)

        # Final layer
        outputs = GATv2Conv(output_dim, heads=1)([x, edge_input])
        outputs = keras.layers.Activation("softmax")(outputs)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_residual_connections_model(self, sample_graph_data):
        """Test model with residual connections between layers."""
        data = sample_graph_data
        hidden_dim = data["input_dim"]  # Same dim for residual connections
        output_dim = 12

        # Build model with residual connections
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        # First layer
        x1 = GCNConv(hidden_dim)([node_input, edge_input])
        x1 = keras.layers.Activation("relu")(x1)

        # Second layer with residual
        x2 = GCNConv(hidden_dim)([x1, edge_input])
        x2 = keras.layers.Activation("relu")(x2)
        x2_residual = keras.layers.Add()([x1, x2])  # Residual connection
        x2_residual = keras.layers.Activation("relu")(x2_residual)

        # Third layer with residual
        x3 = GCNConv(hidden_dim)([x2_residual, edge_input])
        x3 = keras.layers.Activation("relu")(x3)
        x3_residual = keras.layers.Add()([x2_residual, x3])  # Residual connection

        # Output layer
        outputs = keras.layers.Dense(output_dim, activation="relu")(x3_residual)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Test forward pass
        predictions = model([data["node_features"], data["edge_indices"]])

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_model_training_step(self, sample_graph_data):
        """Test that integrated models can perform training steps."""
        data = sample_graph_data
        hidden_dim = 16
        num_classes = 3

        # Build simple model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        x = GCNConv(hidden_dim)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Compile model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Create dummy labels
        labels = np.random.randint(0, num_classes, size=(data["num_nodes"],))

        # Test training step (should not crash)
        history = model.fit(
            [data["node_features"], data["edge_indices"]], labels, epochs=1, verbose=0
        )

        assert len(history.history["loss"]) == 1
        assert history.history["loss"][0] > 0  # Loss should be positive

    def test_model_serialization(self, sample_graph_data):
        """Test that integrated models can be saved and loaded."""
        data = sample_graph_data
        hidden_dim = 8
        output_dim = 4

        # Build model
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2, data["num_edges"]), name="edge_indices")

        x = GCNConv(hidden_dim)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(output_dim)(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Get predictions before serialization
        pred_before = model([data["node_features"], data["edge_indices"]])

        # Test config serialization/deserialization
        config = model.get_config()
        model_from_config = keras.Model.from_config(config)

        # Verify the models have the same structure
        assert len(model.layers) == len(model_from_config.layers)

        # Set same weights and test predictions
        model_from_config.set_weights(model.get_weights())
        pred_after = model_from_config([data["node_features"], data["edge_indices"]])

        np.testing.assert_allclose(pred_before.numpy(), pred_after.numpy(), rtol=1e-6)
