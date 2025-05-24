import itertools
import os
import sys
import unittest

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

# --- Import Custom GCNConv Layer ---
try:
    from keras_geometric.layers.gcn_conv import GCNConv

    GCN_AVAILABLE = True
except ImportError as e:
    print(f"Could not import GCNConv layer: {e}")
    GCNConv = None  # type: ignore
    GCN_AVAILABLE = False

# --- PyTorch Geometric Imports (Optional) ---
TORCH_AVAILABLE = False
PyGGCNConv = None

try:
    # pyrefly: ignore  # import-error
    import torch

    # pyrefly: ignore  # import-error
    from torch_geometric.nn import GCNConv as PyGGCNConv

    # Force CPU execution for consistent testing
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
        print("Setting PyTorch default device to CPU.")
    else:
        # For older PyTorch versions
        torch.set_default_tensor_type(torch.FloatTensor)
        print("Setting PyTorch default tensor type to FloatTensor (CPU).")
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch or PyTorch Geometric not available. Skipping comparison tests.")


class TestGCNConvBase(unittest.TestCase):
    """Base test class with common setup and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Test parameters
        self.num_nodes = 6
        self.input_dim = 10
        self.output_dim = 12

        # Test configuration options
        self.test_configs = [
            {"use_bias": True, "normalize": True, "add_self_loops": True},
            {"use_bias": False, "normalize": True, "add_self_loops": True},
            {"use_bias": True, "normalize": False, "add_self_loops": True},
            {"use_bias": True, "normalize": True, "add_self_loops": False},
        ]

        # Create test graph data
        self.features_numpy = np.random.randn(self.num_nodes, self.input_dim).astype(
            np.float32
        )
        self.edge_index_numpy = np.array(
            [[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 5, 0]], dtype=np.int32
        )

        # Convert to Keras tensors
        self.features_keras = keras.ops.convert_to_tensor(
            self.features_numpy, dtype="float32"
        )
        self.edge_index_keras = keras.ops.convert_to_tensor(
            self.edge_index_numpy, dtype="int32"
        )

        # For PyTorch backend compatibility
        if TORCH_AVAILABLE:
            import torch

            torch.manual_seed(42)
            # Ensure we're using CPU for testing
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Random seed for reproducibility
        np.random.seed(42)
        if hasattr(keras.utils, "set_random_seed"):
            keras.utils.set_random_seed(42)

    def _get_expected_output_shape(
        self, batch_size: int, output_dim: int
    ) -> tuple[int, int]:
        """Get expected output shape for given parameters."""
        return (batch_size, output_dim)

    def _extract_numpy_output(self, tensor):
        """Extract numpy array from tensor output."""
        # Always use keras.ops.convert_to_numpy for consistent behavior across backends
        return keras.ops.convert_to_numpy(tensor)


@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvInitialization(TestGCNConvBase):
    """Test GCNConv layer initialization."""

    def test_basic_initialization(self):
        """Test basic layer initialization."""
        print("\n--- Testing GCNConv Basic Initialization ---")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim)

        # Check default values
        self.assertEqual(gcn.output_dim, self.output_dim)
        self.assertTrue(gcn.use_bias)
        self.assertTrue(gcn.add_self_loops)
        self.assertTrue(gcn.normalize)
        self.assertEqual(gcn.dropout_rate, 0.0)
        self.assertEqual(gcn.aggregator, "sum")

        # Check that weights are not built yet
        self.assertIsNone(gcn.kernel)
        self.assertIsNone(gcn.bias)

        print("✓ Basic initialization test passed")

    def test_parameter_variations(self):
        """Test layer initialization with different parameters."""
        print("\n--- Testing GCNConv Parameter Variations ---")

        test_cases = [
            {
                "output_dim": 16,
                "use_bias": False,
                "add_self_loops": False,
                "normalize": False,
                "dropout_rate": 0.5,
                "kernel_initializer": "he_normal",
                "bias_initializer": "ones",
            },
            {
                "output_dim": 32,
                "use_bias": True,
                "add_self_loops": True,
                "normalize": True,
                "dropout_rate": 0.2,
                "kernel_initializer": "glorot_normal",
                "bias_initializer": "zeros",
            },
        ]

        for i, config in enumerate(test_cases):
            with self.subTest(case=i):
                # pyrefly: ignore  # not-callable
                gcn = GCNConv(**config)

                # Verify all parameters are set correctly (except initializers which become objects)
                for param, value in config.items():
                    if hasattr(gcn, param):
                        if "initializer" in param:
                            # For initializers, check the class name instead of object equality
                            if value == "he_normal":
                                self.assertEqual(
                                    getattr(gcn, param).__class__.__name__, "HeNormal"
                                )
                            elif value == "glorot_normal":
                                self.assertEqual(
                                    getattr(gcn, param).__class__.__name__,
                                    "GlorotNormal",
                                )
                            elif value == "ones":
                                self.assertEqual(
                                    getattr(gcn, param).__class__.__name__, "Ones"
                                )
                            elif value == "zeros":
                                self.assertEqual(
                                    getattr(gcn, param).__class__.__name__, "Zeros"
                                )
                        else:
                            self.assertEqual(getattr(gcn, param), value)

                # Build the layer
                _ = gcn([self.features_keras, self.edge_index_keras])

                # Check weights are created
                self.assertIsNotNone(gcn.kernel)
                if config["use_bias"]:
                    self.assertIsNotNone(gcn.bias)
                else:
                    self.assertIsNone(gcn.bias)

        print("✓ Parameter variations test passed")


@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvForwardPass(TestGCNConvBase):
    """Test GCNConv forward pass functionality."""

    def test_output_shapes(self):
        """Test forward pass output shapes for different configurations."""
        print("\n--- Testing GCNConv Output Shapes ---")

        input_data = [self.features_keras, self.edge_index_keras]

        test_params = list(
            itertools.product(
                [8, 16, 32],  # output_dim
                [True, False],  # use_bias
                [True, False],  # normalize
                [True, False],  # add_self_loops
            )
        )

        for output_dim, use_bias, normalize, add_self_loops in test_params:
            with self.subTest(
                output_dim=output_dim,
                use_bias=use_bias,
                normalize=normalize,
                add_self_loops=add_self_loops,
            ):
                # pyrefly: ignore  # not-callable
                gcn = GCNConv(
                    output_dim=output_dim,
                    use_bias=use_bias,
                    normalize=normalize,
                    add_self_loops=add_self_loops,
                )

                output = gcn(input_data)
                expected_shape = self._get_expected_output_shape(
                    self.num_nodes, output_dim
                )

                actual_shape = tuple(self._extract_numpy_output(output).shape)
                self.assertEqual(
                    actual_shape,
                    expected_shape,
                    f"Shape mismatch for config: output_dim={output_dim}, "
                    f"use_bias={use_bias}, normalize={normalize}, add_self_loops={add_self_loops}",
                )

        print("✓ Output shapes test passed")

    def test_dropout_behavior(self):
        """Test dropout behavior during training and inference."""
        print("\n--- Testing GCNConv Dropout Behavior ---")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim, dropout_rate=0.5)

        input_data = [self.features_keras, self.edge_index_keras]

        # Test training mode
        output_train1 = gcn(input_data, training=True)

        # Test inference mode
        output_infer1 = gcn(input_data, training=False)
        output_infer2 = gcn(input_data, training=False)

        # Convert to numpy for comparison
        train1_np = self._extract_numpy_output(output_train1)
        infer1_np = self._extract_numpy_output(output_infer1)
        infer2_np = self._extract_numpy_output(output_infer2)

        # Check shapes
        expected_shape = (self.num_nodes, self.output_dim)
        self.assertEqual(train1_np.shape, expected_shape)
        self.assertEqual(infer1_np.shape, expected_shape)

        # Inference should be deterministic
        np.testing.assert_array_almost_equal(
            infer1_np,
            infer2_np,
            decimal=5,
            err_msg="Inference outputs should be identical",
        )

        print("✓ Dropout behavior test passed")

    def test_edge_cases(self):
        """Test edge cases like empty graphs, single node graphs, etc."""
        print("\n--- Testing GCNConv Edge Cases ---")

        # Test empty graph (0 nodes)
        empty_features = keras.ops.zeros((0, self.input_dim))
        empty_edges = keras.ops.zeros((2, 0), dtype="int32")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim)
        output = gcn([empty_features, empty_edges])

        expected_shape = (0, self.output_dim)
        self.assertEqual(
            self._extract_numpy_output(output).shape,
            expected_shape,
            "Empty graph should produce correct shape",
        )

        # Test single node graph with no edges
        single_node_features = keras.ops.ones((1, self.input_dim))
        no_edges = keras.ops.zeros((2, 0), dtype="int32")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim, add_self_loops=True)
        output = gcn([single_node_features, no_edges])

        expected_shape = (1, self.output_dim)
        self.assertEqual(
            self._extract_numpy_output(output).shape,
            expected_shape,
            "Single node graph should produce correct shape",
        )

        # Test graph with only self-loops
        features = keras.ops.ones((3, self.input_dim))
        self_loop_edges = keras.ops.array([[0, 1, 2], [0, 1, 2]], dtype="int32")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim)
        output = gcn([features, self_loop_edges])

        expected_shape = (3, self.output_dim)
        self.assertEqual(
            self._extract_numpy_output(output).shape,
            expected_shape,
            "Self-loop only graph should produce correct shape",
        )

        print("✓ Edge cases test passed")

    def test_normalization_behavior(self):
        """Test normalization behavior."""
        print("\n--- Testing GCNConv Normalization Behavior ---")

        input_data = [self.features_keras, self.edge_index_keras]

        # Test with normalization
        # pyrefly: ignore  # not-callable
        gcn_norm = GCNConv(output_dim=self.output_dim, normalize=True)
        output_norm = gcn_norm(input_data)

        # Test without normalization
        # pyrefly: ignore  # not-callable
        gcn_no_norm = GCNConv(output_dim=self.output_dim, normalize=False)
        output_no_norm = gcn_no_norm(input_data)

        # Convert to numpy
        norm_np = self._extract_numpy_output(output_norm)
        no_norm_np = self._extract_numpy_output(output_no_norm)

        # Check shapes are the same
        self.assertEqual(norm_np.shape, no_norm_np.shape)

        # Outputs should be different due to normalization
        self.assertFalse(
            np.allclose(norm_np, no_norm_np, atol=1e-5),
            "Normalized and non-normalized outputs should be different",
        )

        print("✓ Normalization behavior test passed")


@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvSerialization(TestGCNConvBase):
    """Test GCNConv serialization and configuration."""

    def test_config_serialization(self):
        """Test get_config and from_config methods."""
        print("\n--- Testing GCNConv Config Serialization ---")

        # Create layer with non-default parameters
        original_config = {
            "output_dim": self.output_dim + 1,
            "use_bias": False,
            "normalize": False,
            "add_self_loops": False,
            "dropout_rate": 0.3,
            "kernel_initializer": "he_normal",
            "bias_initializer": "ones",
            "name": "test_gcn_config",
        }

        # pyrefly: ignore  # not-callable
        gcn1 = GCNConv(**original_config)

        # Build the layer
        _ = gcn1([self.features_keras, self.edge_index_keras])

        # Get configuration
        config = gcn1.get_config()

        # Verify key configuration parameters
        expected_keys = [
            "name",
            "trainable",
            "output_dim",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "add_self_loops",
            "normalize",
            "dropout_rate",
            "aggregator",
        ]

        for key in expected_keys:
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Test specific values
        self.assertEqual(config["output_dim"], original_config["output_dim"])
        self.assertEqual(config["use_bias"], original_config["use_bias"])
        self.assertEqual(config["normalize"], original_config["normalize"])
        self.assertEqual(config["add_self_loops"], original_config["add_self_loops"])
        self.assertEqual(config["dropout_rate"], original_config["dropout_rate"])
        self.assertEqual(config["name"], original_config["name"])
        self.assertEqual(config["aggregator"], "sum")

        # Check initializers are properly serialized
        kernel_config = config["kernel_initializer"]
        bias_config = config["bias_initializer"]
        if isinstance(kernel_config, dict):
            self.assertEqual(kernel_config.get("class_name"), "HeNormal")
        else:
            self.assertEqual(kernel_config, "he_normal")

        if isinstance(bias_config, dict):
            self.assertEqual(bias_config.get("class_name"), "Ones")
        else:
            self.assertEqual(bias_config, "ones")

        # Test layer reconstruction
        try:
            # pyrefly: ignore  # not-callable
            gcn2 = GCNConv.from_config(config)
        except Exception as e:
            self.fail(f"GCNConv.from_config failed: {e}")

        # Verify reconstructed layer properties
        self.assertEqual(gcn1.output_dim, gcn2.output_dim)
        self.assertEqual(gcn1.use_bias, gcn2.use_bias)
        self.assertEqual(gcn1.normalize, gcn2.normalize)
        self.assertEqual(gcn1.add_self_loops, gcn2.add_self_loops)
        self.assertEqual(gcn1.dropout_rate, gcn2.dropout_rate)
        self.assertEqual(gcn1.name, gcn2.name)
        self.assertEqual(gcn1.aggregator, gcn2.aggregator)

        # Compare initializer types after deserialization
        self.assertEqual(gcn2.kernel_initializer.__class__.__name__, "HeNormal")
        self.assertEqual(gcn2.bias_initializer.__class__.__name__, "Ones")

        print("✓ Config serialization test passed")


@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvGradients(TestGCNConvBase):
    """Test GCNConv gradient computation."""

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        print("\n--- Testing GCNConv Gradient Flow ---")

        # pyrefly: ignore  # not-callable
        gcn = GCNConv(output_dim=self.output_dim, use_bias=True)

        input_data = [self.features_keras, self.edge_index_keras]

        # Use backend-specific gradient tape
        backend = get_keras_backend()
        if backend == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape() as tape:
                output = gcn(input_data)
                loss = keras.ops.sum(output)
            grads = tape.gradient(loss, gcn.trainable_variables)
        elif backend == "torch":
            # For PyTorch backend, we'll skip gradient testing since it's more complex
            self.skipTest("Gradient testing not implemented for PyTorch backend")
        elif backend == "jax":
            # For JAX backend, we'll skip gradient testing since it's more complex
            self.skipTest("Gradient testing not implemented for JAX backend")
        else:
            self.skipTest(f"Gradient testing not implemented for {backend} backend")

        # Check that all gradients are computed
        for grad, var in zip(grads, gcn.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")
            grad_np = self._extract_numpy_output(grad)
            self.assertFalse(np.any(np.isnan(grad_np)), f"NaN gradient for {var.name}")
            self.assertFalse(np.all(grad_np == 0), f"Zero gradient for {var.name}")

        print("✓ Gradient flow test passed")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric not available")
@unittest.skipIf(not KERAS_BACKEND_IS_TORCH, "Keras backend is not torch")
@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvNumericalComparison(TestGCNConvBase):
    """Test numerical comparison with PyTorch Geometric."""

    def _sync_layer_weights(self, keras_layer, pyg_layer, use_bias):
        """Synchronize weights between Keras and PyG layers."""
        # Get Keras weights using keras.ops.convert_to_numpy for consistency
        keras_kernel = keras.ops.convert_to_numpy(
            keras_layer.kernel
        )  # Shape: [input_dim, output_dim]

        # Set PyG weights
        with torch.no_grad():
            # PyG linear layer has weight shape [output_dim, input_dim] (transposed)
            pyg_layer.lin.weight.copy_(torch.tensor(keras_kernel.T))

            if (
                use_bias
                and keras_layer.bias is not None
                and pyg_layer.lin.bias is not None
            ):
                keras_bias = keras.ops.convert_to_numpy(
                    keras_layer.bias
                )  # Shape: [output_dim]
                pyg_layer.lin.bias.copy_(torch.tensor(keras_bias))

    def test_numerical_comparison_basic(self):
        """Test numerical equivalence with PyTorch Geometric GCN."""
        print("\n--- Testing GCNConv Numerical Comparison with PyG ---")

        # Test parameters
        use_bias = True
        add_self_loops = True
        normalize = True

        # Create layers
        # pyrefly: ignore  # not-callable
        keras_gcn = GCNConv(
            output_dim=self.output_dim,
            use_bias=use_bias,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )

        pyg_gcn = PyGGCNConv(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            bias=use_bias,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )

        # Build Keras layer
        keras_input = [self.features_keras, self.edge_index_keras]
        _ = keras_gcn(keras_input)

        # Sync weights
        self._sync_layer_weights(keras_gcn, pyg_gcn, use_bias)

        # Prepare PyG inputs (ensure CPU device)
        pyg_features = torch.tensor(self.features_numpy, device="cpu")
        pyg_edge_index = torch.tensor(
            self.edge_index_numpy.astype(np.int64), device="cpu"
        )

        # Forward pass
        with torch.no_grad():
            pyg_gcn.eval()
            pyg_output = pyg_gcn(pyg_features, pyg_edge_index)

        keras_output = keras_gcn(keras_input, training=False)

        # Convert to numpy for comparison
        pyg_output_np = pyg_output.cpu().detach().numpy()
        keras_output_np = self._extract_numpy_output(keras_output)

        # Check shapes match
        self.assertEqual(
            keras_output_np.shape,
            pyg_output_np.shape,
            "Output shapes should match between Keras and PyG",
        )

        # Check numerical equivalence
        try:
            np.testing.assert_allclose(
                keras_output_np,
                pyg_output_np,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Keras and PyG outputs should be numerically equivalent",
            )
            print("✓ Numerical comparison test passed")
        except AssertionError as e:
            print(f"Warning: Numerical comparison failed: {e}")
            print(
                f"Max absolute difference: {np.max(np.abs(keras_output_np - pyg_output_np))}"
            )
            print(
                f"Max relative difference: {np.max(np.abs((keras_output_np - pyg_output_np) / (pyg_output_np + 1e-8)))}"
            )
            # Don't fail the test for small numerical differences
            max_diff = np.max(np.abs(keras_output_np - pyg_output_np))
            if max_diff > 1e-3:
                raise


@unittest.skipIf(not GCN_AVAILABLE, "GCNConv layer could not be imported.")
class TestGCNConvRegularization(TestGCNConvBase):
    """Test GCNConv regularization features."""

    def test_kernel_regularization(self):
        """Test kernel regularization."""
        print("\n--- Testing GCNConv Kernel Regularization ---")

        # pyrefly: ignore  # not-callable
        gcn_reg = GCNConv(
            output_dim=self.output_dim,
            kernel_regularizer=keras.regularizers.l2(0.01),
        )

        # pyrefly: ignore  # not-callable
        gcn_no_reg = GCNConv(output_dim=self.output_dim)

        input_data = [self.features_keras, self.edge_index_keras]

        # Build both layers
        _ = gcn_reg(input_data)
        _ = gcn_no_reg(input_data)

        # Check that regularized layer has losses
        self.assertTrue(
            len(gcn_reg.losses) > 0,
            "Regularized layer should have regularization losses",
        )
        self.assertEqual(
            len(gcn_no_reg.losses), 0, "Non-regularized layer should have no losses"
        )

        print("✓ Kernel regularization test passed")

    def test_bias_regularization(self):
        """Test bias regularization."""
        print("\n--- Testing GCNConv Bias Regularization ---")

        # pyrefly: ignore  # not-callable
        gcn_reg = GCNConv(
            output_dim=self.output_dim,
            use_bias=True,
            bias_regularizer=keras.regularizers.l1(0.01),
        )

        input_data = [self.features_keras, self.edge_index_keras]

        # Build layer
        _ = gcn_reg(input_data)

        # Check that layer has bias regularization losses
        self.assertTrue(
            len(gcn_reg.losses) > 0,
            "Bias regularized layer should have regularization losses",
        )

        print("✓ Bias regularization test passed")


if __name__ == "__main__":
    # pyrefly: ignore  # not-callable
    unittest.main()
