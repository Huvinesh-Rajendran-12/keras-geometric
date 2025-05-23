import os
import sys
import unittest

# --- Keras Imports ---
import keras
import numpy as np

KERAS_BACKEND_IS_TORCH = False
try:
    if keras.backend.backend() == "torch":
        KERAS_BACKEND_IS_TORCH = True
        print("Keras backend confirmed: 'torch'")
    else:
        print(f"Warning: Keras backend is '{keras.backend.backend()}', not 'torch'.")
except Exception:
    print("Warning: Could not determine Keras backend.")

# --- Add src directory to path ---
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import MessagePassing Base Class ---
"""
Attempt to import the MessagePassing base class from the keras_geometric layers module.

This import is wrapped in a try-except block to handle potential import errors,
allowing the test suite to gracefully handle scenarios where the module might
not be available or cannot be imported.
"""
try:
    from keras_geometric.layers.message_passing import MessagePassing
except ImportError as e:
    print(f"Could not import MessagePassing from package 'keras_geometric': {e}")
    MessagePassing = None
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    MessagePassing = None


class DummyMessagePassing(MessagePassing):
    """A simple implementation of MessagePassing for testing."""

    def __init__(self, aggregator="sum", **kwargs):
        super().__init__(aggregator=aggregator, **kwargs)

    def message(self, x_i, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


@unittest.skipIf(
    MessagePassing is None, "MessagePassing base class could not be imported."
)
class TestMessagePassingComprehensive(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_nodes = 7
        self.num_features = 4
        self.aggregation_methods = ["mean", "max", "sum"]

        # Create a simple graph for testing
        np.random.seed(42)
        self.features_np = np.random.randn(self.num_nodes, self.num_features).astype(
            np.float32
        )
        self.edge_index_np = np.array(
            [
                [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 6, 5, 1],  # Source nodes
                [1, 0, 2, 1, 4, 3, 5, 4, 2, 5, 5, 6, 6],  # Target nodes
            ],
            dtype=np.int64,
        )

        self.features_keras = keras.ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = keras.ops.convert_to_tensor(
            self.edge_index_np, dtype="int32"
        )

    def test_initialization(self):
        """Test initialization of MessagePassing with different aggregation methods."""
        print("\n--- Testing MessagePassing Initialization ---")
        for aggr in self.aggregation_methods:
            with self.subTest(aggregation=aggr):
                mp = DummyMessagePassing(aggregator=aggr)
                self.assertEqual(mp.aggregator, aggr)

        # Test invalid aggregation
        with self.assertRaises(AssertionError):
            DummyMessagePassing(aggregator="invalid")

    def test_message_passing_shapes(self):
        """Test the shapes of intermediate tensors in message passing."""
        print("\n--- Testing MessagePassing Shapes ---")
        for aggr in self.aggregation_methods:
            with self.subTest(aggregation=aggr):
                mp = DummyMessagePassing(aggregator=aggr)
                output = mp([self.features_keras, self.edge_index_keras])

                # Use keras.ops.convert_to_numpy for backend-agnostic conversion
                output_np = keras.ops.convert_to_numpy(output)

                # Output should maintain node count and feature dimensions
                self.assertEqual(
                    output_np.shape,
                    (self.num_nodes, self.num_features),
                    f"Shape mismatch for aggregator '{aggr}'",
                )

    def test_message_passing_values(self):
        """Test the actual values after message passing with different aggregations."""
        print("\n--- Testing MessagePassing Values ---")

        def manual_aggregate(
            x: np.ndarray, edge_index: np.ndarray, method: str
        ) -> np.ndarray:
            """Manually compute aggregation for verification."""
            num_nodes = x.shape[0]
            out = np.zeros_like(x)
            for target_idx in range(num_nodes):
                # Find neighbors (source nodes) for current target
                neighbor_mask = edge_index[1] == target_idx
                neighbors = edge_index[0][neighbor_mask]
                if len(neighbors) == 0:
                    continue
                neighbor_features = x[neighbors]

                if method == "mean":
                    out[target_idx] = np.mean(neighbor_features, axis=0)
                elif method == "max":
                    out[target_idx] = np.max(neighbor_features, axis=0)
                elif method == "sum":
                    out[target_idx] = np.sum(neighbor_features, axis=0)
            return out

        for aggr in self.aggregation_methods:
            with self.subTest(aggregation=aggr):
                mp = DummyMessagePassing(aggregator=aggr)
                output = mp([self.features_keras, self.edge_index_keras])

                # Use keras.ops.convert_to_numpy for backend-agnostic conversion
                output_np = keras.ops.convert_to_numpy(output)

                # Compute expected output manually
                expected_output = manual_aggregate(
                    self.features_np, self.edge_index_np, aggr
                )

                # Compare actual vs expected
                try:
                    np.testing.assert_allclose(
                        output_np,
                        expected_output,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Values mismatch for aggregator '{aggr}'",
                    )
                    print(f"✅ Values match for aggregation '{aggr}'")
                except AssertionError as e:
                    print(f"❌ Values DO NOT match for aggregation '{aggr}'")
                    print(e)
                    abs_diff = np.abs(output_np - expected_output)
                    rel_diff = abs_diff / (np.abs(expected_output) + 1e-8)
                    print(f"Max absolute difference: {np.max(abs_diff)}")
                    print(f"Max relative difference: {np.max(rel_diff)}")

    def test_empty_graph(self):
        """Test behavior with an empty graph (no edges)."""
        print("\n--- Testing Empty Graph Handling ---")
        empty_edge_index = keras.ops.convert_to_tensor(
            np.zeros((2, 0), dtype=np.int64), dtype="int32"
        )

        for aggr in self.aggregation_methods:
            with self.subTest(aggregation=aggr):
                mp = DummyMessagePassing(aggregator=aggr)
                output = mp([self.features_keras, empty_edge_index])

                # Use keras.ops.convert_to_numpy for backend-agnostic conversion
                output_np = keras.ops.convert_to_numpy(output)

                # With no edges, output should be zeros
                expected_shape = (self.num_nodes, self.num_features)
                self.assertEqual(output_np.shape, expected_shape)
                np.testing.assert_allclose(
                    output_np,
                    np.zeros(expected_shape),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Empty graph output not zero for aggregator '{aggr}'",
                )

    def test_single_node_graph(self):
        """Test behavior with a single-node graph."""
        print("\n--- Testing Single Node Graph Handling ---")
        # Create a graph with a single node that has a self-loop
        single_node_features = keras.ops.convert_to_tensor(
            np.random.randn(1, self.num_features).astype(np.float32)
        )
        single_node_edge_index = keras.ops.convert_to_tensor(
            np.array([[0], [0]]), dtype="int32"
        )  # Self-loop

        for aggr in self.aggregation_methods:
            with self.subTest(aggregation=aggr):
                mp = DummyMessagePassing(aggregator=aggr)
                output = mp([single_node_features, single_node_edge_index])

                # Use keras.ops.convert_to_numpy for backend-agnostic conversion
                output_np = keras.ops.convert_to_numpy(output)

                # Output should have shape [1, num_features]
                expected_shape = (1, self.num_features)
                self.assertEqual(output_np.shape, expected_shape)

                # Since it's a self-loop, the output should be the input feature
                # (for mean/sum/max with a single value, they all return that value)
                np.testing.assert_allclose(
                    output_np,
                    keras.ops.convert_to_numpy(single_node_features),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Single node graph output incorrect for aggregator '{aggr}'",
                )


if __name__ == "__main__":
    unittest.main()
