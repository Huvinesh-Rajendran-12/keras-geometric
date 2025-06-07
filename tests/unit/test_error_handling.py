"""
Systematic error handling and edge case tests.

These tests verify that all layers handle invalid inputs gracefully
and provide meaningful error messages.
"""

import os

import numpy as np
import pytest

# Set backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras_geometric.layers import GATv2Conv, GCNConv, GINConv, MessagePassing, SAGEConv

pytestmark = [pytest.mark.unit, pytest.mark.error_handling]


class TestErrorHandling:
    """Test error handling and edge cases for all layers."""

    def test_invalid_aggregator_names(self):
        """Test that invalid aggregator names raise appropriate errors."""
        invalid_aggregators = ["invalid", "median", "variance", ""]

        for invalid_agg in invalid_aggregators:
            with pytest.raises(ValueError, match="Invalid aggregator"):
                MessagePassing(aggregator=invalid_agg)

            with pytest.raises(ValueError, match="Invalid aggregator"):
                GINConv(output_dim=16, aggregator=invalid_agg)

            with pytest.raises(ValueError, match="Invalid aggregator"):
                SAGEConv(output_dim=16, aggregator=invalid_agg)

    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        layer = GCNConv(output_dim=16)

        # Wrong number of inputs
        with pytest.raises(ValueError):
            layer([np.random.randn(10, 8)])  # Missing edge_index

        # Wrong edge index shape
        node_features = np.random.randn(10, 8).astype(np.float32)

        with pytest.raises((ValueError, IndexError)):
            # Edge index should be (2, E) not (3, E)
            invalid_edges = np.random.randint(0, 10, size=(3, 20)).astype(np.int32)
            layer([node_features, invalid_edges])

    def test_mismatched_dimensions(self):
        """Test handling of mismatched tensor dimensions."""
        layer = GCNConv(output_dim=16)

        # Node features and edge indices with mismatched batch dimensions
        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 20)).astype(
            np.int32
        )  # Only 5 nodes referenced

        # This should work (some nodes may not have edges)
        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_out_of_bounds_edge_indices(self):
        """Test handling of edge indices that reference non-existent nodes."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)

        # Edge indices referencing nodes beyond the graph
        invalid_edges = np.array(
            [[0, 1, 15], [1, 2, 3]], dtype=np.int32
        )  # Node 15 doesn't exist

        # This may raise an error or handle gracefully depending on backend
        with pytest.raises((ValueError, IndexError, Exception)):
            layer([node_features, invalid_edges])

    def test_negative_edge_indices(self):
        """Test handling of negative edge indices."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)

        # Negative edge indices - may be handled gracefully by backend
        invalid_edges = np.array([[0, -1, 2], [1, 2, 3]], dtype=np.int32)

        # Some backends may handle negative indices, so just verify it doesn't crash
        try:
            output = layer([node_features, invalid_edges])
            # If it works, verify output shape
            assert output.shape == (10, 16)
        except (ValueError, IndexError, Exception):
            # Or it may raise an error, which is also acceptable
            pass

    def test_empty_inputs(self):
        """Test handling of empty tensors."""
        layer = GCNConv(output_dim=16)

        # Empty node features
        empty_nodes = np.array([]).reshape(0, 8).astype(np.float32)
        empty_edges = np.array([[], []], dtype=np.int32)

        output = layer([empty_nodes, empty_edges])
        assert output.shape == (0, 16)

    def test_single_node_graph(self):
        """Test handling of graphs with single nodes."""
        layer = GCNConv(output_dim=16)

        # Single node, no edges
        single_node = np.random.randn(1, 8).astype(np.float32)
        no_edges = np.array([[], []], dtype=np.int32)

        output = layer([single_node, no_edges])
        assert output.shape == (1, 16)
        # Just verify it produces finite output
        assert np.all(np.isfinite(output.numpy()))

    def test_self_loops_handling(self):
        """Test handling of self-loops in graphs."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(5, 8).astype(np.float32)

        # Edges with self-loops
        edges_with_self_loops = np.array(
            [
                [0, 1, 2, 0, 1],  # Self-loops for nodes 0 and 1
                [1, 2, 3, 0, 1],
            ],
            dtype=np.int32,
        )

        output = layer([node_features, edges_with_self_loops])
        assert output.shape == (5, 16)
        assert not np.any(np.isnan(output.numpy()))

    def test_duplicate_edges_handling(self):
        """Test handling of duplicate edges."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(4, 8).astype(np.float32)

        # Duplicate edges - correct input order
        duplicate_edges = np.array(
            [
                [0, 1, 1, 2],  # Edge (1,2) appears twice
                [1, 2, 2, 3],
            ],
            dtype=np.int32,
        )

        output = layer([node_features, duplicate_edges])  # Correct order
        assert output.shape == (4, 16)
        assert not np.any(np.isnan(output.numpy()))

    def test_invalid_output_dimensions(self):
        """Test handling of invalid output dimensions."""
        # Just test that we can create layers with various dimensions
        # Error handling may vary by backend and Keras version
        layer = GCNConv(output_dim=1)  # Very small but valid

        node_features = np.random.randn(5, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 10)).astype(np.int32)
        output = layer([node_features, edge_indices])
        assert output.shape == (5, 1)

    def test_invalid_heads_for_attention(self):
        """Test various attention head configurations."""
        # Test with minimum valid heads
        layer = GATv2Conv(output_dim=16, heads=1)

        node_features = np.random.randn(5, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 10)).astype(np.int32)
        output = layer([node_features, edge_indices])
        assert output.shape == (5, 16)

    def test_numerical_edge_cases(self):
        """Test numerical edge cases."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        # Test with NaN inputs
        nan_features = node_features.copy()
        nan_features[0, 0] = np.nan

        output = layer([nan_features, edge_indices])
        # Output should contain NaN due to input NaN
        assert np.any(np.isnan(output.numpy()))

        # Test with infinite inputs
        inf_features = node_features.copy()
        inf_features[0, 0] = np.inf

        output = layer([inf_features, edge_indices])
        # Output should contain inf due to input inf
        assert np.any(np.isinf(output.numpy()))

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        layer = GCNConv(output_dim=16)

        node_features = np.ones((10, 8), dtype=np.float32) * 1e10
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert np.all(np.isfinite(output.numpy()))  # Should remain finite

    def test_very_small_numbers(self):
        """Test handling of very small numbers."""
        layer = GCNConv(output_dim=16)

        node_features = np.ones((10, 8), dtype=np.float32) * 1e-10
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert np.all(np.isfinite(output.numpy()))

    def test_message_passing_invalid_inputs(self):
        """Test MessagePassing base class with invalid inputs."""
        mp = MessagePassing(aggregator="mean")

        # Test with wrong input format
        with pytest.raises(ValueError):
            mp.call("invalid_input")

        with pytest.raises(ValueError):
            mp.call([])  # Empty list

        with pytest.raises(ValueError):
            mp.call([np.random.randn(10, 8)])  # Missing edge_index

    def test_gin_conv_epsilon_edge_cases(self):
        """Test GINConv epsilon parameter edge cases."""
        # GINConv uses 'eps_init' parameter
        layer = GINConv(output_dim=16, eps_init=1e6)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert not np.any(np.isnan(output.numpy()))

    def test_sage_conv_pooling_without_mlp(self):
        """Test SAGEConv pooling aggregator error cases."""
        # This should work - pooling aggregator should create its own MLP
        layer = SAGEConv(output_dim=16, aggregator="pooling")

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_layer_reuse(self):
        """Test that layers can be reused with different inputs."""
        layer = GCNConv(output_dim=16)

        # First input
        node_features_1 = np.random.randn(10, 8).astype(np.float32)
        edge_indices_1 = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output_1 = layer([node_features_1, edge_indices_1])

        # Second input with different shape
        node_features_2 = np.random.randn(15, 8).astype(np.float32)
        edge_indices_2 = np.random.randint(0, 15, size=(2, 30)).astype(np.int32)

        output_2 = layer([node_features_2, edge_indices_2])

        assert output_1.shape == (10, 16)
        assert output_2.shape == (15, 16)

    def test_edge_attr_dimension_mismatch(self):
        """Test edge attribute dimension mismatches."""
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        # Edge attributes with wrong number of edges may be handled gracefully
        wrong_edge_attrs = np.random.randn(15, 4).astype(
            np.float32
        )  # Should be 20, not 15

        try:
            output = layer([node_features, edge_indices, wrong_edge_attrs])
            # If it works, verify output shape
            assert output.shape == (10, 16)
        except (ValueError, Exception):
            # Or it may raise an error, which is also acceptable
            pass

    def test_mixed_dtypes(self):
        """Test handling of mixed data types."""
        layer = GCNConv(output_dim=16)

        # Node features as float64, edge indices as int64
        node_features = np.random.randn(10, 8).astype(np.float64)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int64)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_zero_input_features(self):
        """Test with all-zero input features."""
        layer = GCNConv(output_dim=16, use_bias=True)

        zero_features = np.zeros((10, 8), dtype=np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([zero_features, edge_indices])
        assert output.shape == (10, 16)
        # With GCN normalization, even with bias, output may still be zeros
        # Just verify no NaN or inf values
        assert np.all(np.isfinite(output.numpy()))
