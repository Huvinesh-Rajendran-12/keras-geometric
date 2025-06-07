"""
Backend compatibility tests for keras-geometric.

These tests ensure that all GNN layers work correctly across different
Keras backends (TensorFlow, PyTorch, JAX).
"""

import importlib
import os
import sys
from typing import Any

import keras
import numpy as np
import pytest

from keras_geometric.layers import GATv2Conv, GCNConv, GINConv, MessagePassing, SAGEConv

pytestmark = pytest.mark.backend

# Available backends to test
AVAILABLE_BACKENDS = []

# Check which backends are available
if importlib.util.find_spec("tensorflow") is not None:
    AVAILABLE_BACKENDS.append("tensorflow")

if importlib.util.find_spec("torch") is not None:
    AVAILABLE_BACKENDS.append("torch")

if importlib.util.find_spec("jax") is not None:
    AVAILABLE_BACKENDS.append("jax")


def switch_backend(backend_name: str) -> None:
    """Switch Keras backend and reload keras_geometric modules."""
    # Set environment variable
    os.environ["KERAS_BACKEND"] = backend_name

    # Clear keras from cache
    modules_to_remove = [
        name for name in sys.modules.keys() if name.startswith("keras")
    ]
    for module in modules_to_remove:
        sys.modules.pop(module, None)

    # Clear keras_geometric from cache
    kg_modules = [
        name for name in sys.modules.keys() if name.startswith("keras_geometric")
    ]
    for module in kg_modules:
        sys.modules.pop(module, None)

    # Reimport keras to pick up new backend
    importlib.reload(keras)


class TestBackendCompatibility:
    """Test that all layers work correctly across different backends."""

    @pytest.fixture
    def sample_data(self):
        """Create sample graph data for testing."""
        num_nodes = 20
        num_edges = 60
        input_dim = 8

        # Use fixed seed for reproducible tests
        np.random.seed(42)

        node_features = np.random.randn(num_nodes, input_dim).astype(np.float32)
        edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(
            np.int32
        )

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "num_nodes": num_nodes,
            "input_dim": input_dim,
        }

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_gcn_conv_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test GCNConv works across all backends."""
        # Skip if backend not available
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        # Switch to backend
        switch_backend(backend)

        # Reimport after backend switch

        # Create layer
        layer = GCNConv(output_dim=16, use_bias=True)

        # Test forward pass
        output = layer([sample_data["node_features"], sample_data["edge_indices"]])

        # Verify output properties
        assert output.shape == (sample_data["num_nodes"], 16)
        assert not np.any(np.isnan(output.numpy()))
        assert np.all(output.numpy() >= 0)  # ReLU should be non-negative

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_gatv2_conv_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test GATv2Conv works across all backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Test single-head attention
        layer = GATv2Conv(output_dim=12, heads=1)
        output = layer([sample_data["node_features"], sample_data["edge_indices"]])

        assert output.shape == (sample_data["num_nodes"], 12)
        assert not np.any(np.isnan(output.numpy()))

        # Test multi-head attention
        layer_multi = GATv2Conv(output_dim=8, heads=4)
        output_multi = layer_multi(
            [sample_data["node_features"], sample_data["edge_indices"]]
        )

        assert output_multi.shape == (sample_data["num_nodes"], 8)
        assert not np.any(np.isnan(output_multi.numpy()))

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("aggregator", ["mean", "max", "sum", "min", "std"])
    def test_gin_conv_aggregators_backend_compatibility(
        self, backend: str, aggregator: str, sample_data: dict[str, Any]
    ):
        """Test GINConv with different aggregators across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create layer with specific aggregator
        layer = GINConv(output_dim=10, aggregator=aggregator)
        output = layer([sample_data["node_features"], sample_data["edge_indices"]])

        assert output.shape == (sample_data["num_nodes"], 10)
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("aggregator", ["mean", "max", "sum"])
    def test_sage_conv_aggregators_backend_compatibility(
        self, backend: str, aggregator: str, sample_data: dict[str, Any]
    ):
        """Test SAGEConv with different aggregators across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create layer with specific aggregator
        layer = SAGEConv(output_dim=14, aggregator=aggregator)
        output = layer([sample_data["node_features"], sample_data["edge_indices"]])

        assert output.shape == (sample_data["num_nodes"], 14)
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_message_passing_base_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test MessagePassing base class across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create simple MessagePassing layer
        layer = MessagePassing(aggregator="mean")

        # Test propagate method
        output = layer.propagate(
            x=sample_data["node_features"], edge_index=sample_data["edge_indices"]
        )

        assert output.shape == sample_data["node_features"].shape
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_gradient_computation_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test gradient computation works across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create model
        node_input = keras.Input(shape=(sample_data["input_dim"],))
        edge_input = keras.Input(shape=(2, sample_data["edge_indices"].shape[1]))

        x = GCNConv(16)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Create dummy targets
        targets = np.random.randint(0, 3, size=(sample_data["num_nodes"],))

        # Test gradient computation by running one training step
        with keras.utils.custom_object_scope():
            initial_loss = model.evaluate(
                [sample_data["node_features"], sample_data["edge_indices"]],
                targets,
                verbose=0,
            )

            model.fit(
                [sample_data["node_features"], sample_data["edge_indices"]],
                targets,
                epochs=1,
                verbose=0,
            )

            final_loss = model.evaluate(
                [sample_data["node_features"], sample_data["edge_indices"]],
                targets,
                verbose=0,
            )

        # Loss should be finite
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_serialization_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test model serialization works across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create and test layer
        layer = GCNConv(output_dim=8)

        # Get config
        config = layer.get_config()

        # Recreate layer from config
        layer_from_config = GCNConv.from_config(config)

        # Verify configs match
        assert layer.get_config() == layer_from_config.get_config()

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_numerical_stability_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test numerical stability across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create layer
        layer = GCNConv(output_dim=16)

        # Test with very small values
        small_features = sample_data["node_features"] * 1e-6
        output_small = layer([small_features, sample_data["edge_indices"]])

        assert not np.any(np.isnan(output_small.numpy()))
        assert np.all(np.isfinite(output_small.numpy()))

        # Test with large values
        large_features = sample_data["node_features"] * 1e3
        output_large = layer([large_features, sample_data["edge_indices"]])

        assert not np.any(np.isnan(output_large.numpy()))
        assert np.all(np.isfinite(output_large.numpy()))

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_empty_graph_backend_compatibility(
        self, backend: str, sample_data: dict[str, Any]
    ):
        """Test handling of empty graphs across backends."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")

        switch_backend(backend)

        # Create layer
        layer = GCNConv(output_dim=8)

        # Test with empty edges
        empty_edges = np.array([[], []], dtype=np.int32)
        output = layer([sample_data["node_features"], empty_edges])

        assert output.shape == (sample_data["num_nodes"], 8)
        assert np.allclose(output.numpy(), 0.0)  # Should be zeros

    def test_cross_backend_numerical_consistency(self, sample_data: dict[str, Any]):
        """Test that results are numerically consistent across backends."""
        if len(AVAILABLE_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend testing")

        results = {}

        # Test each available backend
        for backend in AVAILABLE_BACKENDS[:2]:  # Test first 2 available backends
            switch_backend(backend)

            # Create layer with fixed seed
            np.random.seed(42)
            layer = GCNConv(output_dim=16, use_bias=False)  # No bias for consistency

            # Get output
            output = layer([sample_data["node_features"], sample_data["edge_indices"]])
            results[backend] = output.numpy()

        # Compare results between backends
        backend_names = list(results.keys())
        if len(backend_names) >= 2:
            # Results should be similar (allowing for backend differences)
            np.testing.assert_allclose(
                results[backend_names[0]],
                results[backend_names[1]],
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Results differ between {backend_names[0]} and {backend_names[1]}",
            )
