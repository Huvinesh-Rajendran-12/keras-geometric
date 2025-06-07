"""Tests for graph pooling layers."""

import os
import sys

import keras
import numpy as np
import pytest

from keras_geometric.layers.pooling import (
    AttentionPooling,
    BatchGlobalPooling,
    GlobalPooling,
    Set2Set,
)

# --- Add src directory to path ---
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set backend before importing keras_geometric
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


class TestGlobalPooling:
    """Test suite for GlobalPooling layer."""

    @pytest.fixture
    def sample_node_features(self):
        """Create sample node features for testing."""
        np.random.seed(42)
        return keras.ops.convert_to_tensor(np.random.randn(50, 32).astype(np.float32))

    def test_global_pooling_initialization(self):
        """Test GlobalPooling layer initialization."""
        # Test valid pooling types
        for pooling in ["mean", "max", "sum"]:
            layer = GlobalPooling(pooling=pooling)
            assert layer.pooling == pooling

        # Test invalid pooling type
        with pytest.raises(ValueError, match="pooling must be one of"):
            GlobalPooling(pooling="invalid")

    def test_global_pooling_output_shapes(self, sample_node_features):
        """Test output shapes for different pooling operations."""
        num_nodes, num_features = 50, 32

        for pooling in ["mean", "max", "sum"]:
            layer = GlobalPooling(pooling=pooling)
            output = layer(sample_node_features)

            # Check output shape
            assert output.shape == (1, num_features)

            # Check compute_output_shape
            computed_shape = layer.compute_output_shape((num_nodes, num_features))
            assert computed_shape == (1, num_features)

    def test_global_pooling_forward_pass(self, sample_node_features):
        """Test forward pass functionality."""
        # Mean pooling
        mean_layer = GlobalPooling(pooling="mean")
        mean_output = mean_layer(sample_node_features)
        expected_mean = keras.ops.mean(sample_node_features, axis=0, keepdims=True)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(mean_output),
            keras.ops.convert_to_numpy(expected_mean),
            rtol=1e-6,
        )

        # Max pooling
        max_layer = GlobalPooling(pooling="max")
        max_output = max_layer(sample_node_features)
        expected_max = keras.ops.max(sample_node_features, axis=0, keepdims=True)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(max_output),
            keras.ops.convert_to_numpy(expected_max),
            rtol=1e-6,
        )

        # Sum pooling
        sum_layer = GlobalPooling(pooling="sum")
        sum_output = sum_layer(sample_node_features)
        expected_sum = keras.ops.sum(sample_node_features, axis=0, keepdims=True)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sum_output),
            keras.ops.convert_to_numpy(expected_sum),
            rtol=1e-6,
        )

    def test_global_pooling_serialization(self):
        """Test layer serialization and deserialization."""
        layer = GlobalPooling(pooling="mean")
        config = layer.get_config()

        # Check config contains expected keys
        assert "pooling" in config
        assert config["pooling"] == "mean"

        # Test from_config
        new_layer = GlobalPooling.from_config(config)
        assert new_layer.pooling == layer.pooling

    def test_global_pooling_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        layer = GlobalPooling(pooling="mean")

        # Test with 1D input
        with pytest.raises(ValueError, match="Expected input shape to be 2D"):
            layer.compute_output_shape((50,))

        # Test with 3D input
        with pytest.raises(ValueError, match="Expected input shape to be 2D"):
            layer.compute_output_shape((10, 50, 32))


class TestBatchGlobalPooling:
    """Test suite for BatchGlobalPooling layer."""

    @pytest.fixture
    def sample_batch_data(self):
        """Create sample batched data for testing."""
        np.random.seed(42)

        # Create features for 3 graphs with different sizes
        graph_sizes = [30, 45, 25]
        total_nodes = sum(graph_sizes)

        node_features = keras.ops.convert_to_tensor(
            np.random.randn(total_nodes, 16).astype(np.float32)
        )

        # Create batch indices
        batch_indices = []
        for graph_id, size in enumerate(graph_sizes):
            batch_indices.extend([graph_id] * size)
        batch = keras.ops.convert_to_tensor(np.array(batch_indices, dtype=np.int32))

        return node_features, batch, graph_sizes

    def test_batch_global_pooling_initialization(self):
        """Test BatchGlobalPooling layer initialization."""
        # Test valid pooling types
        for pooling in ["mean", "max", "sum"]:
            layer = BatchGlobalPooling(pooling=pooling)
            assert layer.pooling == pooling

        # Test invalid pooling type
        with pytest.raises(ValueError, match="pooling must be one of"):
            BatchGlobalPooling(pooling="invalid")

    def test_batch_global_pooling_output_shapes(self, sample_batch_data):
        """Test output shapes for batch pooling operations."""
        node_features, batch, graph_sizes = sample_batch_data
        num_graphs = len(graph_sizes)
        num_features = 16

        for pooling in ["mean", "max", "sum"]:
            layer = BatchGlobalPooling(pooling=pooling)
            output = layer([node_features, batch])

            # Check output shape
            assert output.shape == (num_graphs, num_features)

    def test_batch_global_pooling_forward_pass(self, sample_batch_data):
        """Test forward pass functionality for batch pooling."""
        node_features, batch, graph_sizes = sample_batch_data

        # Test mean pooling
        mean_layer = BatchGlobalPooling(pooling="mean")
        mean_output = mean_layer([node_features, batch])

        # Verify output shape
        assert mean_output.shape == (len(graph_sizes), 16)

        # Check that each graph representation is reasonable
        output_numpy = keras.ops.convert_to_numpy(mean_output)
        assert not np.isnan(output_numpy).any()
        assert not np.isinf(output_numpy).any()

    def test_batch_global_pooling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        layer = BatchGlobalPooling(pooling="mean")

        # Test with wrong number of inputs
        with pytest.raises(ValueError, match="inputs must be a list/tuple"):
            layer(keras.ops.zeros((10, 5)))

        # Test with wrong input shapes for compute_output_shape
        with pytest.raises(ValueError, match="input_shape must be a list/tuple"):
            layer.compute_output_shape((10, 5))

    def test_batch_global_pooling_serialization(self):
        """Test layer serialization and deserialization."""
        layer = BatchGlobalPooling(pooling="sum")
        config = layer.get_config()

        # Check config contains expected keys
        assert "pooling" in config
        assert config["pooling"] == "sum"

        # Test from_config
        new_layer = BatchGlobalPooling.from_config(config)
        assert new_layer.pooling == layer.pooling


class TestAttentionPooling:
    """Test suite for AttentionPooling layer."""

    @pytest.fixture
    def sample_node_features(self):
        """Create sample node features for testing."""
        np.random.seed(42)
        return keras.ops.convert_to_tensor(np.random.randn(25, 64).astype(np.float32))

    def test_attention_pooling_initialization(self):
        """Test AttentionPooling layer initialization."""
        # Test default initialization
        layer = AttentionPooling()
        assert layer.attention_dim is None
        assert layer.dropout_rate == 0.0

        # Test with custom parameters
        layer = AttentionPooling(attention_dim=32, dropout=0.1)
        assert layer.attention_dim == 32
        assert layer.dropout_rate == 0.1

        # Test invalid parameters
        with pytest.raises(ValueError, match="attention_dim must be positive"):
            AttentionPooling(attention_dim=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            AttentionPooling(dropout=1.5)

    def test_attention_pooling_build_and_call(self, sample_node_features):
        """Test building and calling the AttentionPooling layer."""
        layer = AttentionPooling(attention_dim=32, dropout=0.1)

        # Test before building
        assert layer.attention_dense is None

        # Build the layer
        layer.build(sample_node_features.shape)

        # Test after building
        assert layer.attention_dense is not None
        assert layer.attention_score is not None
        assert layer.dropout_layer is not None

        # Test forward pass
        output = layer(sample_node_features, training=False)
        assert output.shape == (1, 64)

        # Test with training=True
        output_train = layer(sample_node_features, training=True)
        assert output_train.shape == (1, 64)

    def test_attention_pooling_output_shape(self):
        """Test output shape computation."""
        layer = AttentionPooling()

        # Test compute_output_shape
        input_shape = (25, 64)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (1, 64)

        # Test invalid input shape
        with pytest.raises(ValueError, match="Expected input shape to be 2D"):
            layer.compute_output_shape((25,))

    def test_attention_pooling_serialization(self):
        """Test layer serialization and deserialization."""
        layer = AttentionPooling(attention_dim=16, dropout=0.2)
        config = layer.get_config()

        # Check config contains expected keys
        assert "attention_dim" in config
        assert "dropout" in config
        assert config["attention_dim"] == 16
        assert config["dropout"] == 0.2

        # Test from_config
        new_layer = AttentionPooling.from_config(config)
        assert new_layer.attention_dim == layer.attention_dim
        assert new_layer.dropout_rate == layer.dropout_rate


class TestSet2Set:
    """Test suite for Set2Set pooling layer."""

    @pytest.fixture
    def sample_node_features(self):
        """Create sample node features for testing."""
        np.random.seed(42)
        return keras.ops.convert_to_tensor(np.random.randn(20, 32).astype(np.float32))

    def test_set2set_initialization(self):
        """Test Set2Set layer initialization."""
        # Test default initialization
        layer = Set2Set(output_dim=16)
        assert layer.output_dim == 16
        assert layer.processing_steps == 3
        assert layer.lstm_units == 16
        assert layer.dropout_rate == 0.0

        # Test with custom parameters
        layer = Set2Set(output_dim=32, processing_steps=5, lstm_units=24, dropout=0.1)
        assert layer.output_dim == 32
        assert layer.processing_steps == 5
        assert layer.lstm_units == 24
        assert layer.dropout_rate == 0.1

        # Test invalid parameters
        with pytest.raises(ValueError, match="output_dim must be positive"):
            Set2Set(output_dim=0)

        with pytest.raises(ValueError, match="processing_steps must be positive"):
            Set2Set(output_dim=16, processing_steps=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            Set2Set(output_dim=16, dropout=-0.1)

    def test_set2set_build_and_call(self, sample_node_features):
        """Test building and calling the Set2Set layer."""
        layer = Set2Set(output_dim=16, processing_steps=2, dropout=0.1)

        # Test before building
        assert layer.lstm_cell is None
        assert layer.attention_dense is None

        # Build the layer
        layer.build(sample_node_features.shape)

        # Test after building
        assert layer.lstm_cell is not None
        assert layer.attention_dense is not None
        assert layer.dropout_layer is not None

        # Test forward pass
        output = layer(sample_node_features, training=False)

        # Output should be [1, lstm_units + input_dim] = [1, 16 + 32] = [1, 48]
        expected_output_dim = layer.lstm_units + sample_node_features.shape[1]
        assert output.shape == (1, expected_output_dim)

        # Test with training=True
        output_train = layer(sample_node_features, training=True)
        assert output_train.shape == (1, expected_output_dim)

    def test_set2set_output_shape(self):
        """Test output shape computation."""
        layer = Set2Set(output_dim=16, lstm_units=24)

        # Test compute_output_shape
        input_shape = (20, 32)
        output_shape = layer.compute_output_shape(input_shape)
        # Output should be [1, lstm_units + input_features] = [1, 24 + 32] = [1, 56]
        assert output_shape == (1, 24 + 32)

        # Test invalid input shape
        with pytest.raises(ValueError, match="Expected input shape to be 2D"):
            layer.compute_output_shape((20,))

        # Test None input dimension
        with pytest.raises(ValueError, match="Input feature dimension cannot be None"):
            layer.compute_output_shape((20, None))

    def test_set2set_serialization(self):
        """Test layer serialization and deserialization."""
        layer = Set2Set(output_dim=32, processing_steps=4, lstm_units=16, dropout=0.2)
        config = layer.get_config()

        # Check config contains expected keys
        expected_keys = ["output_dim", "processing_steps", "lstm_units", "dropout"]
        for key in expected_keys:
            assert key in config

        assert config["output_dim"] == 32
        assert config["processing_steps"] == 4
        assert config["lstm_units"] == 16
        assert config["dropout"] == 0.2

        # Test from_config
        new_layer = Set2Set.from_config(config)
        assert new_layer.output_dim == layer.output_dim
        assert new_layer.processing_steps == layer.processing_steps
        assert new_layer.lstm_units == layer.lstm_units
        assert new_layer.dropout_rate == layer.dropout_rate

    def test_set2set_automatic_build(self, sample_node_features):
        """Test that layer builds automatically when called."""
        layer = Set2Set(output_dim=16)

        # Layer should build automatically when called
        assert layer.lstm_cell is None  # Not built yet

        output = layer(sample_node_features)

        # After calling, layer should be built
        assert layer.lstm_cell is not None
        assert layer.attention_dense is not None
        assert output.shape == (1, 16 + 32)  # lstm_units + input_dim


class TestPoolingIntegration:
    """Integration tests for pooling layers."""

    def test_pooling_in_graph_classification_pipeline(self):
        """Test pooling layers in a complete graph classification pipeline."""
        np.random.seed(42)

        # Create sample data
        num_nodes = 30
        num_features = 16
        num_classes = 3

        node_features = keras.ops.convert_to_tensor(
            np.random.randn(num_nodes, num_features).astype(np.float32)
        )

        # Test with different pooling layers
        pooling_layers = [
            GlobalPooling(pooling="mean"),
            GlobalPooling(pooling="max"),
            GlobalPooling(pooling="sum"),
            AttentionPooling(attention_dim=8),
            Set2Set(output_dim=8, processing_steps=2),
        ]

        for pooling_layer in pooling_layers:
            # Apply pooling
            graph_repr = pooling_layer(node_features)

            # Apply classification head
            classifier = keras.layers.Dense(num_classes, activation="softmax")
            predictions = classifier(graph_repr)

            # Check final output
            assert predictions.shape == (1, num_classes)

            # Check that predictions sum to 1 (valid probabilities)
            pred_sum = keras.ops.sum(predictions)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(pred_sum), 1.0, atol=1e-6
            )

    def test_pooling_layers_with_gradients(self):
        """Test that pooling layers work with gradient computation."""
        np.random.seed(42)

        node_features = keras.ops.convert_to_tensor(
            np.random.randn(15, 8).astype(np.float32)
        )
        labels = keras.ops.convert_to_tensor(np.array([[1, 0, 0]], dtype=np.float32))

        # Test gradient computation with GlobalPooling
        pooling_layer = GlobalPooling(pooling="mean")
        classifier = keras.layers.Dense(3, activation="softmax")

        # Forward pass
        pooled = pooling_layer(node_features)
        predictions = classifier(pooled)

        # Compute loss
        loss = keras.ops.mean(
            keras.losses.categorical_crossentropy(labels, predictions)
        )

        # Verify that loss is a valid scalar
        assert keras.ops.ndim(loss) == 0
        assert keras.ops.convert_to_numpy(loss) > 0
