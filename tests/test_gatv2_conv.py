import itertools
import os
import sys
import unittest

# --- Keras Imports ---
import keras
import numpy as np
from keras import ops


# --- Backend Detection ---
def get_keras_backend():
    """Get the current Keras backend safely."""
    try:
        return keras.backend.backend()
    except Exception:
        return "unknown"


KERAS_BACKEND = get_keras_backend()
KERAS_BACKEND_IS_TORCH = KERAS_BACKEND == "torch"

if KERAS_BACKEND_IS_TORCH:
    print("Keras backend confirmed: 'torch'")
else:
    print(
        f"Warning: Keras backend is '{KERAS_BACKEND}', not 'torch'. "
        "Numerical comparison tests will be skipped."
    )


# --- Add src directory to path ---
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import Custom GATv2Conv Layer ---
try:
    from keras_geometric.layers.gatv2_conv import GATv2Conv

    GATV2_AVAILABLE = True
except ImportError as e:
    print(f"Could not import GATv2Conv layer: {e}")
    GATv2Conv = None
    GATV2_AVAILABLE = False

# --- PyTorch Geometric Imports (Optional) ---
TORCH_AVAILABLE = False
PyGGATv2Conv = None
torch = None

try:
    import torch
    from torch_geometric.nn import GATv2Conv as PyGGATv2Conv

    # Force CPU execution for consistent testing
    torch.set_default_device("cpu")
    print("Setting PyTorch default device to CPU.")
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch or PyTorch Geometric not available. Skipping comparison tests.")


class TestGATv2ConvBase(unittest.TestCase):
    """Base test class with common setup and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Test parameters
        self.num_nodes = 6
        self.input_dim = 10
        self.output_dim = 12

        # Test configuration options
        self.heads_options = [1, 3]
        self.concat_options = [True, False]
        self.bias_options = [True, False]
        self.selfloop_options = [True, False]

        # Set random seeds for reproducibility
        np.random.seed(44)
        if TORCH_AVAILABLE:
            torch.manual_seed(44)

        # Generate test data
        self.features_np = np.random.randn(self.num_nodes, self.input_dim).astype(
            np.float32
        )
        self.edge_index_np = np.array(
            [
                [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 5, 1],
                [1, 0, 2, 1, 4, 3, 5, 4, 2, 5, 0, 0],
            ],
            dtype=np.int64,
        )

        # Convert to Keras tensors using ops
        self.features_keras = ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = ops.convert_to_tensor(self.edge_index_np, dtype="int32")

        # Convert to PyTorch tensors if available
        if TORCH_AVAILABLE:
            self.features_torch = torch.tensor(self.features_np)
            self.edge_index_torch = torch.tensor(self.edge_index_np)

    def _get_expected_output_shape(self, heads: int, concat: bool) -> tuple[int, int]:
        """Calculate expected output shape based on configuration."""
        if concat:
            return (self.num_nodes, self.output_dim * heads)
        else:
            return (self.num_nodes, self.output_dim)

    def _extract_numpy_output(self, output) -> np.ndarray:
        """Extract numpy array from output tensor (backend agnostic)."""
        return ops.convert_to_numpy(output)


@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvInitialization(TestGATv2ConvBase):
    """Test GATv2Conv layer initialization."""

    def test_basic_initialization(self):
        """Test basic layer initialization."""
        print("\n--- Testing GATv2Conv Basic Initialization ---")

        gat = GATv2Conv(output_dim=self.output_dim)

        # Check default values
        self.assertEqual(gat.output_dim, self.output_dim)
        self.assertEqual(gat.heads, 1)
        self.assertTrue(gat.concat)
        self.assertEqual(gat.negative_slope, 0.2)
        self.assertEqual(gat.dropout_rate, 0.0)
        self.assertTrue(gat.use_bias)
        self.assertTrue(gat.add_self_loops_flag)  # Updated attribute name
        self.assertEqual(gat.aggregator, "sum")

    def test_parameter_variations(self):
        """Test layer initialization with various parameter combinations."""
        print("\n--- Testing GATv2Conv Parameter Variations ---")

        test_params = list(
            itertools.product(
                self.heads_options,
                self.concat_options,
                self.bias_options,
                self.selfloop_options,
            )
        )

        for heads, concat, use_bias, add_loops in test_params:
            with self.subTest(
                heads=heads, concat=concat, bias=use_bias, loops=add_loops
            ):
                gat = GATv2Conv(
                    output_dim=self.output_dim,
                    heads=heads,
                    concat=concat,
                    use_bias=use_bias,
                    add_self_loops=add_loops,
                    dropout=0.1,
                    negative_slope=0.1,
                )

                # Verify parameters
                self.assertEqual(gat.output_dim, self.output_dim)
                self.assertEqual(gat.heads, heads)
                self.assertEqual(gat.concat, concat)
                self.assertEqual(gat.use_bias, use_bias)
                self.assertEqual(gat.add_self_loops_flag, add_loops)  # Updated
                self.assertEqual(gat.dropout_rate, 0.1)
                self.assertEqual(gat.negative_slope, 0.1)


@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvForwardPass(TestGATv2ConvBase):
    """Test GATv2Conv forward pass functionality."""

    def test_output_shapes(self):
        """Test forward pass output shapes for different configurations."""
        print("\n--- Testing GATv2Conv Output Shapes ---")

        input_data = [self.features_keras, self.edge_index_keras]

        test_params = list(
            itertools.product(
                self.heads_options,
                self.concat_options,
                self.bias_options,
                self.selfloop_options,
            )
        )

        for heads, concat, use_bias, add_loops in test_params:
            with self.subTest(
                heads=heads, concat=concat, bias=use_bias, loops=add_loops
            ):
                gat = GATv2Conv(
                    output_dim=self.output_dim,
                    heads=heads,
                    concat=concat,
                    use_bias=use_bias,
                    add_self_loops=add_loops,
                )

                output = gat(input_data)
                expected_shape = self._get_expected_output_shape(heads, concat)
                output_shape = tuple(output.shape)

                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Shape mismatch for config: heads={heads}, concat={concat}, "
                    f"bias={use_bias}, loops={add_loops}",
                )

    def test_method_signatures(self):
        """Test that overridden methods have correct signatures."""
        print("\n--- Testing GATv2Conv Method Signatures ---")

        gat = GATv2Conv(output_dim=self.output_dim)

        # Build the layer
        _ = gat([self.features_keras, self.edge_index_keras])

        # Test call method with edge_attr parameter
        output_with_edge_attr = gat(
            [self.features_keras, self.edge_index_keras], edge_attr=None, training=False
        )
        self.assertIsNotNone(output_with_edge_attr)

        # Test that message method accepts all required parameters
        # This is tested internally during forward pass

    def test_dropout_behavior(self):
        """Test dropout behavior during training vs inference."""
        print("\n--- Testing GATv2Conv Dropout Behavior ---")

        # Use high dropout rate to make effects visible
        gat = GATv2Conv(
            output_dim=self.output_dim, heads=1, concat=True, dropout=0.5, use_bias=True
        )

        input_data = [self.features_keras, self.edge_index_keras]

        # Test training mode - outputs should vary due to dropout
        training_outputs = []
        for _ in range(3):
            output = gat(input_data, training=True)
            training_outputs.append(self._extract_numpy_output(output))

        # Verify training outputs differ
        outputs_differ = any(
            not np.allclose(training_outputs[0], output, rtol=1e-5, atol=1e-5)
            for output in training_outputs[1:]
        )
        self.assertTrue(outputs_differ, "Training outputs should differ due to dropout")

        # Test inference mode - outputs should be consistent
        inference_outputs = []
        for _ in range(3):
            output = gat(input_data, training=False)
            inference_outputs.append(self._extract_numpy_output(output))

        # Verify inference outputs are consistent
        for i in range(1, len(inference_outputs)):
            np.testing.assert_allclose(
                inference_outputs[0],
                inference_outputs[i],
                rtol=1e-5,
                atol=1e-5,
                err_msg="Inference outputs should be identical",
            )

    def test_edge_cases(self):
        """Test edge cases like empty graphs, single node graphs, etc."""
        print("\n--- Testing GATv2Conv Edge Cases ---")

        # Test empty graph (0 nodes)
        empty_features = ops.zeros((0, self.input_dim))
        empty_edges = ops.zeros((2, 0), dtype="int32")

        gat = GATv2Conv(output_dim=self.output_dim, heads=2, concat=True)
        output = gat([empty_features, empty_edges])

        expected_shape = (0, self.output_dim * 2)
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            "Empty graph should produce correct shape",
        )

        # Test single node graph with no edges
        single_node_features = ops.ones((1, self.input_dim))
        no_edges = ops.zeros((2, 0), dtype="int32")

        gat = GATv2Conv(
            output_dim=self.output_dim, heads=2, concat=False, add_self_loops=True
        )
        output = gat([single_node_features, no_edges])

        expected_shape = (1, self.output_dim)
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            "Single node graph should produce correct shape",
        )

        # Test graph with only self-loops
        features = ops.ones((3, self.input_dim))
        self_loop_edges = ops.convert_to_tensor([[0, 1, 2], [0, 1, 2]], dtype="int32")

        gat = GATv2Conv(output_dim=self.output_dim, heads=1, concat=True)
        output = gat([features, self_loop_edges])

        expected_shape = (3, self.output_dim)
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            "Self-loop only graph should produce correct shape",
        )

    def test_bipartite_graphs(self):
        """Test GATv2Conv with bipartite graphs."""
        print("\n--- Testing GATv2Conv with Bipartite Graphs ---")

        # Create bipartite graph data
        source_features = ops.ones((4, self.input_dim))
        target_features = ops.ones((3, self.input_dim))

        # Edges from source (0-3) to target (0-2)
        bipartite_edges = ops.convert_to_tensor(
            [[0, 1, 2, 3], [0, 1, 2, 0]], dtype="int32"
        )

        gat = GATv2Conv(
            output_dim=self.output_dim,
            heads=2,
            concat=True,
            add_self_loops=False,  # No self-loops for bipartite
        )

        # Test with bipartite input
        output = gat.propagate(
            x=(target_features, source_features), edge_index=bipartite_edges
        )

        expected_shape = (3, self.output_dim * 2)  # Target nodes, concatenated heads
        self.assertEqual(
            tuple(output.shape),
            expected_shape,
            "Bipartite graph should produce correct shape",
        )


@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvSerialization(TestGATv2ConvBase):
    """Test GATv2Conv serialization and configuration."""

    def test_config_serialization(self):
        """Test get_config and from_config methods."""
        print("\n--- Testing GATv2Conv Config Serialization ---")


@unittest.skipIf(
    not TORCH_AVAILABLE or not PyGGATv2Conv,
    "PyTorch or PyTorch Geometric GATv2Conv not available.",
)
@unittest.skipIf(
    KERAS_BACKEND != "torch",
    "Numerical comparison tests only run with PyTorch backend.",
)
class TestGATv2ConvNumericalComparison(TestGATv2ConvBase):
    """Test GATv2Conv numerical output against PyTorch Geometric."""

    def test_numerical_comparison_basic(self):
        """Compare GATv2Conv output with PyTorch Geometric."""
        print("\n--- Testing GATv2Conv Numerical Comparison ---")

        # Define parameters
        output_dim = 16
        heads = 4
        concat = True
        negative_slope = 0.2
        dropout = 0.0  # No dropout for numerical comparison

        # Create Keras Geometric layer
        kg_gat = GATv2Conv(
            output_dim=output_dim,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            use_bias=True,
            add_self_loops=True,
        )

        # Create PyTorch Geometric layer
        # PyG GATv2Conv expects (in_channels, out_channels)
        pyg_gat = PyGGATv2Conv(
            in_channels=self.input_dim,
            out_channels=output_dim,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=True,
            bias=True,
        )

        # Keras layer builds on first call
        kg_output = kg_gat([self.features_keras, self.edge_index_keras], training=False)

        # Perform a dummy forward pass on PyG layer to ensure weights are built
        # Ensure dropout is off and gradients are disabled for this pass
        pyg_gat.train(False)  # Set to evaluation mode
        with torch.no_grad():
            _ = pyg_gat(self.features_torch, self.edge_index_torch)

        # Transfer weights from Keras to PyG
        # Note: PyG uses separate lin_l and lin_r for target and source nodes
        # while Keras uses a single linear_transform for both

        # Get Keras weights
        kg_kernel = kg_gat.linear_transform.kernel  # [input_dim, heads * output_dim]
        kg_kernel_np = ops.convert_to_numpy(kg_kernel).astype(np.float32)

        # PyG expects [heads * output_dim, input_dim] format
        kg_kernel_transposed = kg_kernel_np.T

        # For GATv2, both lin_l and lin_r should use the same transformation
        # This matches the mathematical formulation where the same W is applied to both h_i and h_j
        pyg_gat.lin_l.weight.data.copy_(torch.from_numpy(kg_kernel_transposed))
        pyg_gat.lin_r.weight.data.copy_(torch.from_numpy(kg_kernel_transposed))

        # Handle linear transformation bias
        if kg_gat.linear_transform.bias is not None:
            kg_bias = kg_gat.linear_transform.bias
            kg_bias_np = ops.convert_to_numpy(kg_bias).astype(np.float32)
            pyg_gat.lin_l.bias.data.copy_(torch.from_numpy(kg_bias_np))
            pyg_gat.lin_r.bias.data.copy_(torch.from_numpy(kg_bias_np))
        else:
            # Zero out bias if Keras doesn't use bias in linear transform
            pyg_gat.lin_l.bias.data.zero_()
            pyg_gat.lin_r.bias.data.zero_()

        # Transfer attention weights
        kg_att = kg_gat.att  # [1, heads, output_dim]
        kg_att_squeezed = ops.squeeze(kg_att, axis=0)  # [heads, output_dim]
        kg_att_np = ops.convert_to_numpy(kg_att_squeezed).astype(np.float32)

        # PyG att parameter should match this shape
        pyg_gat.att.data.copy_(torch.from_numpy(kg_att_np))

        # Handle final bias - this is where Keras and PyG might differ
        if kg_gat.use_bias and kg_gat.bias is not None:
            kg_final_bias = kg_gat.bias
            kg_final_bias_np = ops.convert_to_numpy(kg_final_bias).astype(np.float32)
            pyg_gat.bias.data.copy_(torch.from_numpy(kg_final_bias_np))
        else:
            # Zero out final bias if not used in Keras
            pyg_gat.bias.data.zero_()

        # Run PyTorch Geometric layer (ensure dropout is off for comparison)
        with torch.no_grad():
            pyg_output = pyg_gat(self.features_torch, self.edge_index_torch)

        # Convert PyG output to numpy
        pyg_output_np = pyg_output.detach().cpu().numpy()

        # Convert Keras output to numpy
        kg_output_np = self._extract_numpy_output(kg_output)

        # Compare outputs numerically
        # Use a tolerance for floating point comparisons
        self.assertEqual(
            kg_output_np.shape, pyg_output_np.shape, "Output shapes should match"
        )

        np.testing.assert_allclose(
            kg_output_np,
            pyg_output_np,
            rtol=1e-4,  # Stricter tolerance
            atol=1e-4,
            err_msg="Numerical outputs do not match between Keras and PyG",
        )

        print("✓ Numerical outputs match between Keras and PyG")

        # Create layer with non-default parameters
        original_config = {
            "output_dim": self.output_dim + 1,
            "heads": 4,
            "concat": False,
            "negative_slope": 0.1,
            "dropout": 0.0,
            "use_bias": False,
            "kernel_initializer": "he_normal",
            "bias_initializer": "ones",
            "att_initializer": "glorot_uniform",
            "add_self_loops": False,
            "name": "test_gatv2_config",
        }

        gat1 = GATv2Conv(**original_config)

        # Build the layer
        _ = gat1([self.features_keras, self.edge_index_keras])

        # Get configuration
        config = gat1.get_config()

        # Verify key configuration parameters
        expected_keys = [
            "name",
            "trainable",
            "output_dim",
            "heads",
            "concat",
            "negative_slope",
            "dropout",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "att_initializer",
            "add_self_loops",
            "aggregator",
        ]

        for key in expected_keys:
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Test specific values
        self.assertEqual(config["output_dim"], original_config["output_dim"])
        self.assertEqual(config["heads"], original_config["heads"])
        self.assertEqual(config["concat"], original_config["concat"])
        self.assertEqual(config["negative_slope"], original_config["negative_slope"])
        self.assertEqual(config["dropout"], original_config["dropout"])
        self.assertEqual(config["use_bias"], original_config["use_bias"])
        self.assertEqual(config["add_self_loops"], original_config["add_self_loops"])
        self.assertEqual(config["name"], original_config["name"])
        self.assertEqual(config["aggregator"], "sum")

        # Test layer reconstruction
        try:
            gat2 = GATv2Conv.from_config(config)
        except Exception as e:
            self.fail(f"GATv2Conv.from_config failed: {e}")

        # Verify reconstructed layer properties
        self.assertEqual(gat1.output_dim, gat2.output_dim)
        self.assertEqual(gat1.heads, gat2.heads)
        self.assertEqual(gat1.concat, gat2.concat)
        self.assertEqual(gat1.negative_slope, gat2.negative_slope)
        self.assertEqual(gat1.dropout_rate, gat2.dropout_rate)
        self.assertEqual(gat1.use_bias, gat2.use_bias)
        self.assertEqual(gat1.add_self_loops_flag, gat2.add_self_loops_flag)
        self.assertEqual(gat1.name, gat2.name)


@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvAttentionMechanism(TestGATv2ConvBase):
    """Test GATv2Conv attention mechanism specifics."""

    def test_attention_computation(self):
        """Test that attention is computed correctly."""
        print("\n--- Testing GATv2Conv Attention Computation ---")

        # Create a small graph where we can verify attention
        features = ops.convert_to_tensor(
            [[1.0] * self.input_dim, [0.0] * self.input_dim], dtype="float32"
        )
        edges = ops.convert_to_tensor([[0, 1], [1, 0]], dtype="int32")

        gat = GATv2Conv(
            output_dim=self.output_dim, heads=1, concat=True, add_self_loops=False
        )

        output = gat([features, edges])

        # Verify output shape
        self.assertEqual(tuple(output.shape), (2, self.output_dim))

        # Verify output is not all zeros or NaN
        output_np = self._extract_numpy_output(output)
        self.assertFalse(np.all(output_np == 0))
        self.assertFalse(np.any(np.isnan(output_np)))

    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        print("\n--- Testing Multi-Head Attention ---")

        for concat in [True, False]:
            with self.subTest(concat=concat):
                gat = GATv2Conv(output_dim=self.output_dim, heads=4, concat=concat)

                output = gat([self.features_keras, self.edge_index_keras])

                if concat:
                    expected_shape = (self.num_nodes, self.output_dim * 4)
                else:
                    expected_shape = (self.num_nodes, self.output_dim)

                self.assertEqual(tuple(output.shape), expected_shape)


# Test runner
if __name__ == "__main__":
    # pyrefly: ignore
    unittest.main(verbosity=2)
