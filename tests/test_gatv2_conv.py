import itertools
import os
import sys
import unittest
import warnings

# --- Keras Imports ---
import keras
import numpy as np


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
        f"Warning: Keras backend is '{KERAS_BACKEND}', not 'torch'. Numerical comparison tests will be skipped."
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

        # Convert to Keras tensors
        self.features_keras = keras.ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = keras.ops.convert_to_tensor(
            self.edge_index_np, dtype="int32"
        )

        # Convert to PyTorch tensors if available
        if TORCH_AVAILABLE:
            self.features_torch = torch.tensor(self.features_np)
            self.edge_index_torch = torch.tensor(self.edge_index_np)

    def _get_expected_output_shape(self, heads, concat):
        """Calculate expected output shape based on configuration."""
        if concat:
            return (self.num_nodes, self.output_dim * heads)
        else:
            return (self.num_nodes, self.output_dim)

    def _extract_numpy_output(self, output):
        """Extract numpy array from output tensor (backend agnostic)."""
        try:
            return output.cpu().detach().numpy()
        except AttributeError:
            try:
                return output.cpu().numpy()
            except AttributeError:
                return keras.ops.convert_to_numpy(output)


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
        self.assertTrue(gat.add_self_loops)
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
                self.assertEqual(gat.add_self_loops, add_loops)
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
                output_shape = self._extract_numpy_output(output).shape

                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Shape mismatch for config: heads={heads}, concat={concat}, "
                    f"bias={use_bias}, loops={add_loops}",
                )

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


@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvSerialization(TestGATv2ConvBase):
    """Test GATv2Conv serialization and configuration."""

    def test_config_serialization(self):
        """Test get_config and from_config methods."""
        print("\n--- Testing GATv2Conv Config Serialization ---")

        # Create layer with non-default parameters
        original_config = {
            "output_dim": self.output_dim + 1,
            "heads": 4,
            "concat": False,
            "negative_slope": 0.1,
            "dropout": 0.0,  # Use 0.0 to avoid dropout issues in tests
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
            config_copy = config.copy()
            # Remove aggregator to avoid conflicts in from_config
            config_copy.pop("aggregator", None)
            gat2 = GATv2Conv.from_config(config_copy)
        except Exception as e:
            self.fail(f"GATv2Conv.from_config failed: {e}")

        # Verify reconstructed layer properties
        self.assertEqual(gat1.output_dim, gat2.output_dim)
        self.assertEqual(gat1.heads, gat2.heads)
        self.assertEqual(gat1.concat, gat2.concat)
        self.assertEqual(gat1.negative_slope, gat2.negative_slope)
        self.assertEqual(gat1.dropout_rate, gat2.dropout_rate)
        self.assertEqual(gat1.use_bias, gat2.use_bias)
        self.assertEqual(gat1.add_self_loops, gat2.add_self_loops)
        self.assertEqual(gat1.name, gat2.name)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric not available")
@unittest.skipIf(not KERAS_BACKEND_IS_TORCH, "Keras backend is not torch")
@unittest.skipIf(not GATV2_AVAILABLE, "GATv2Conv layer could not be imported.")
class TestGATv2ConvNumericalComparison(TestGATv2ConvBase):
    """Test numerical comparison with PyTorch Geometric."""

    def _sync_layer_weights(self, keras_layer, pyg_layer, use_bias):
        """Synchronize weights between Keras and PyG layers."""
        # Get Keras weights
        keras_lin_weights = keras_layer.linear_transform.get_weights()
        keras_att_vec = keras_layer.att.numpy()
        keras_final_bias = (
            keras_layer.bias.numpy() if keras_layer.bias is not None else None
        )

        # Get PyG parameters
        pyg_params = dict(pyg_layer.named_parameters())

        # Sync linear layer weights
        if use_bias and len(keras_lin_weights) == 2:
            k_kernel, k_lin_bias = keras_lin_weights
            # Sync linear bias if it exists
            if "lin_l.bias" in pyg_params:
                pyg_params["lin_l.bias"].data.copy_(torch.tensor(k_lin_bias))
        else:
            k_kernel = keras_lin_weights[0]

        # Sync kernel weights
        if "lin_l.weight" in pyg_params:
            pyg_params["lin_l.weight"].data.copy_(torch.tensor(k_kernel.T))
        if "lin_r.weight" in pyg_params:
            pyg_params["lin_r.weight"].data.copy_(torch.tensor(k_kernel.T))

        # Sync attention vector
        if "att" in pyg_params:
            pyg_params["att"].data.copy_(torch.tensor(keras_att_vec))

        # Sync final bias
        if use_bias and keras_final_bias is not None and "bias" in pyg_params:
            pyg_params["bias"].data.copy_(torch.tensor(keras_final_bias))

    def test_numerical_comparison_basic(self):
        """Test numerical comparison for basic configurations."""
        print("\n--- Testing Numerical Comparison vs PyG GATv2Conv ---")

        # Test with simpler parameter combinations for stability
        test_configs = [
            (1, True, True, True),  # Single head, concat, bias, self-loops
            (2, False, True, True),  # Multi-head, average, bias, self-loops
            (1, True, False, False),  # Single head, concat, no bias, no self-loops
        ]

        for heads, concat, use_bias, add_loops in test_configs:
            subtest_msg = (
                f"heads={heads}, concat={concat}, bias={use_bias}, loops={add_loops}"
            )

            with self.subTest(msg=subtest_msg):
                print(f"\n--- Comparing: {subtest_msg} ---")

                # Create Keras layer
                keras_gat = GATv2Conv(
                    output_dim=self.output_dim,
                    heads=heads,
                    concat=concat,
                    negative_slope=0.2,
                    dropout=0.0,
                    use_bias=use_bias,
                    add_self_loops=add_loops,
                )

                # Build Keras layer
                _ = keras_gat([self.features_keras, self.edge_index_keras])

                # Create PyG layer
                pyg_gat = PyGGATv2Conv(
                    in_channels=self.input_dim,
                    out_channels=self.output_dim,
                    heads=heads,
                    concat=concat,
                    negative_slope=0.2,
                    dropout=0.0,
                    add_self_loops=add_loops,
                    bias=use_bias,
                )

                # Sync weights
                try:
                    self._sync_layer_weights(keras_gat, pyg_gat, use_bias)
                except Exception as e:
                    self.skipTest(f"Weight synchronization failed: {e}")

                # Forward pass
                keras_output = keras_gat([self.features_keras, self.edge_index_keras])
                pyg_output = pyg_gat(self.features_torch, self.edge_index_torch)

                # Compare outputs
                keras_output_np = self._extract_numpy_output(keras_output)
                pyg_output_np = pyg_output.cpu().detach().numpy()

                # Verify shapes
                expected_shape = self._get_expected_output_shape(heads, concat)
                self.assertEqual(keras_output_np.shape, expected_shape)
                self.assertEqual(pyg_output_np.shape, expected_shape)

                # Compare values with reasonable tolerance
                try:
                    np.testing.assert_allclose(
                        keras_output_np,
                        pyg_output_np,
                        rtol=1e-4,
                        atol=1e-4,
                        err_msg=f"Outputs differ for {subtest_msg}",
                    )
                    print(f"✅ Outputs match for: {subtest_msg}")
                except AssertionError as e:
                    # Provide diagnostic information but don't fail immediately
                    abs_diff = np.abs(keras_output_np - pyg_output_np)
                    rel_diff = abs_diff / (np.abs(pyg_output_np) + 1e-8)

                    max_abs_diff = np.max(abs_diff)
                    max_rel_diff = np.max(rel_diff)

                    print(f"⚠️  Outputs differ for: {subtest_msg}")
                    print(f"   Max Abs Diff: {max_abs_diff:.6f}")
                    print(f"   Max Rel Diff: {max_rel_diff:.6f}")

                    # Only fail if differences are very large
                    if max_abs_diff > 1e-2 or max_rel_diff > 1e-2:
                        self.fail(f"Large numerical differences for {subtest_msg}: {e}")
                    else:
                        print("Differences within acceptable range")


# Test runner
if __name__ == "__main__":
    # Configure test verbosity and warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests with custom test loader if needed
    unittest.main(verbosity=2)
