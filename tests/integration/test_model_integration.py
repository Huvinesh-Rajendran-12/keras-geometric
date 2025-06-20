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
        """
        Generates synthetic graph data for integration tests.

        Returns:
            A dictionary containing random node features, edge indices, edge attributes,
            and metadata such as the number of nodes, edges, and input feature dimension.
        """
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
        """
        Tests a multi-layer GCN model for node classification using stacked GCNConv layers.

        Builds a three-layer GCN model with ReLU activations and a softmax output, then verifies that the output shape matches the expected number of nodes and classes, and that the softmax outputs sum to 1 for each node.
        """
        data = sample_graph_data
        hidden_dim = 32
        output_dim = 7  # Number of classes

        # Create layers
        gcn1 = GCNConv(hidden_dim, use_bias=True)
        gcn2 = GCNConv(hidden_dim, use_bias=True)
        gcn3 = GCNConv(output_dim, use_bias=True)

        # Forward pass through layers
        x = gcn1([data["node_features"], data["edge_indices"]])
        x = keras.ops.relu(x)

        x = gcn2([x, data["edge_indices"]])
        x = keras.ops.relu(x)

        x = gcn3([x, data["edge_indices"]])
        predictions = keras.ops.softmax(x)

        assert predictions.shape == (data["num_nodes"], output_dim)
        assert np.allclose(
            np.sum(keras.ops.convert_to_numpy(predictions), axis=1), 1.0, atol=1e-6
        )  # Softmax check

    def test_heterogeneous_layer_model(self, sample_graph_data):
        """
        Tests a model that combines GCN, GATv2, and SAGE GNN layers for node feature integration.

        Each layer processes the same node features and edge indices independently. Their outputs are concatenated and passed through a dense layer to produce final node representations. Asserts that the output shape matches the expected dimensions.
        """
        data = sample_graph_data
        hidden_dim = 24
        output_dim = 16

        # Build heterogeneous model - call layers directly instead of using Keras Model
        # GCN layer
        gcn_layer = GCNConv(hidden_dim)
        x1 = gcn_layer([data["node_features"], data["edge_indices"]])
        x1 = keras.ops.relu(x1)

        # GAT layer
        gat_layer = GATv2Conv(hidden_dim, heads=2)
        x2 = gat_layer([data["node_features"], data["edge_indices"]])
        x2 = keras.ops.relu(x2)

        # SAGE layer
        sage_layer = SAGEConv(hidden_dim, aggregator="mean")
        x3 = sage_layer([data["node_features"], data["edge_indices"]])
        x3 = keras.ops.relu(x3)

        # Combine features
        x_combined = keras.ops.concatenate([x1, x2, x3], axis=1)

        # Final layer - combined dim is hidden_dim * 4 = 24 * 4 = 96 (GAT has 2 heads)
        dense_layer = keras.layers.Dense(output_dim, activation="relu")
        predictions = dense_layer(x_combined)

        # Expected combined dim would be hidden_dim * 3 = 24 * 3 = 72, output_dim = 16
        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_graph_classification_model(self, sample_graph_data):
        """
        Tests a graph-level classification model using GINConv layers and mean pooling.

        Builds a model with two GINConv layers, applies mean pooling to obtain a graph embedding, and passes it through a dense softmax classification head. Verifies that the output shape matches a single graph prediction and that the softmax output sums to 1.
        """
        data = sample_graph_data
        hidden_dim = 32
        num_classes = 5

        # Build graph classification model using direct layer calls
        # Node embedding layers
        gin1 = GINConv(hidden_dim, aggregator="sum")
        x = gin1([data["node_features"], data["edge_indices"]])
        x = keras.ops.relu(x)
        x = keras.layers.Dropout(0.3)(x, training=False)

        gin2 = GINConv(hidden_dim, aggregator="sum")
        x = gin2([x, data["edge_indices"]])
        x = keras.ops.relu(x)
        x = keras.layers.Dropout(0.3)(x, training=False)

        # Graph-level pooling (simple mean pooling)
        graph_embedding = keras.ops.mean(x, axis=0, keepdims=True)
        reshape_layer = keras.layers.Reshape((hidden_dim,))
        graph_embedding = reshape_layer(graph_embedding)

        # Classification head
        dense_layer = keras.layers.Dense(num_classes, activation="softmax")
        predictions = dense_layer(graph_embedding)

        assert predictions.shape == (
            1,
            num_classes,
        )  # Single graph prediction with batch dim
        assert np.allclose(
            np.sum(keras.ops.convert_to_numpy(predictions)), 1.0, atol=1e-6
        )  # Softmax check

    def test_sage_with_different_aggregators(self, sample_graph_data):
        """
        Tests a model using SAGEConv layers with mean, max, and sum aggregation strategies.

        Builds a Keras model that applies three SAGEConv layers with different aggregators to the same input, concatenates their outputs, and passes them through a dense layer. Verifies that the model produces outputs of the expected shape.
        """
        data = sample_graph_data
        hidden_dim = 16
        output_dim = 8

        # Build model with different SAGE aggregators
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2,), name="edge_indices")

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
        predictions = model([data["node_features"], data["edge_indices"].T])

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_attention_mechanism_model(self, sample_graph_data):
        """
        Tests a multi-head attention-based GNN model using stacked GATv2Conv layers.

        Builds a model with three GATv2Conv layers employing multiple attention heads and ELU activations, followed by softmax output. Verifies that the final prediction tensor has the expected shape for node classification.
        """
        data = sample_graph_data
        hidden_dim = 20
        output_dim = 10

        # Build attention-based model
        # Multi-head attention layers
        gat1 = GATv2Conv(hidden_dim, heads=4, use_bias=True)
        x = gat1([data["node_features"], data["edge_indices"]])
        x = keras.ops.elu(x)
        x = keras.layers.Dropout(0.1)(x, training=False)

        # Input to this layer is hidden_dim * 4 = 80, so set output_dim to 40 and heads=2 gives 80
        gat2 = GATv2Conv(40, heads=2, use_bias=True)
        x = gat2([x, data["edge_indices"]])
        x = keras.layers.Activation("elu")(x)
        x = keras.layers.Dropout(0.1)(x)

        # Final layer - input is 40 * 2 = 80
        gat3 = GATv2Conv(output_dim, heads=1)
        outputs = gat3([x, data["edge_indices"]])
        predictions = keras.ops.softmax(outputs)

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_residual_connections_model(self, sample_graph_data):
        """
        Tests a GCN-based model with residual connections between layers.

        Builds a multi-layer GCN model using residual connections after the second and third layers, then verifies that the model produces outputs of the expected shape for node-level prediction.
        """
        data = sample_graph_data
        hidden_dim = data["input_dim"]  # Same dim for residual connections
        output_dim = 12

        # Build model with residual connections
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2,), name="edge_indices")

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
        predictions = model([data["node_features"], data["edge_indices"].T])

        assert predictions.shape == (data["num_nodes"], output_dim)

    def test_model_training_step(self, sample_graph_data):
        """
        Tests that a simple integrated GNN model can perform a forward pass and compute a valid categorical crossentropy loss, simulating a training step with gradient computation capability.

        Asserts that the model produces predictions of the correct shape and that the computed loss is positive.
        """
        data = sample_graph_data
        hidden_dim = 16
        num_classes = 3

        # Build simple model layers
        gcn_layer = GCNConv(hidden_dim)
        dense_layer = keras.layers.Dense(num_classes, activation="softmax")

        # Create dummy labels
        labels = np.random.randint(0, num_classes, size=(data["num_nodes"],))
        labels_one_hot = keras.utils.to_categorical(labels, num_classes)

        # Test gradient computation (simulates training step)
        # Forward pass
        x = gcn_layer([data["node_features"], data["edge_indices"]])
        x = keras.ops.relu(x)
        predictions = dense_layer(x)

        # Compute loss
        loss = keras.ops.mean(
            keras.losses.categorical_crossentropy(labels_one_hot, predictions)
        )

        # Verify forward pass works
        assert predictions.shape == (data["num_nodes"], num_classes)
        assert keras.ops.convert_to_numpy(loss) > 0.0  # Loss should be positive

    def test_model_serialization(self, sample_graph_data):
        """
        Tests that a Keras model containing Keras Geometric layers can be serialized and deserialized.
        """
        data = sample_graph_data
        hidden_dim = 8
        output_dim = 4

        # Build model using functional approach with a Keras Geometric layer
        node_input = keras.Input(shape=(data["input_dim"],), name="node_features")
        edge_input = keras.Input(shape=(2,), name="edge_indices")

        # Use a GCNConv layer
        x = GCNConv(hidden_dim)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(output_dim)(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        # Get predictions before serialization
        pred_before = model([data["node_features"], data["edge_indices"].T])

        # Test config serialization/deserialization
        config = model.get_config()
        # Need to register custom layers for deserialization
        custom_objects = {"GCNConv": GCNConv}
        model_from_config = keras.Model.from_config(
            config, custom_objects=custom_objects
        )

        # Verify the models have the same structure
        assert len(model.layers) == len(model_from_config.layers)

        # Set same weights and test predictions
        model_from_config.set_weights(model.get_weights())
        pred_after = model_from_config([data["node_features"], data["edge_indices"].T])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pred_before),
            keras.ops.convert_to_numpy(pred_after),
            rtol=1e-6,
        )
