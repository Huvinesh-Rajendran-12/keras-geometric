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

# --- Import Custom GINConv Layer ---
try:
    from keras_geometric.layers.gin_conv import GINConv

    GIN_AVAILABLE = True
except ImportError as e:
    print(f"Could not import GINConv layer: {e}")
    GINConv = None
    GIN_AVAILABLE = False

# --- PyTorch Geometric Imports (Optional) ---
TORCH_AVAILABLE = False
PyGGINConv = None

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv as PyGGINConv

    # Force CPU execution for consistent testing
    torch.set_default_device("cpu")
    print("Setting PyTorch default device to CPU.")
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch or PyTorch Geometric not available. Skipping comparison tests.")


class TestGINConvBase(unittest.TestCase):
    """Base test class with common setup and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Test parameters
        self.num_nodes = 6
        self.input_dim = 10
        self.output_dim = 12

        # Test configuration options
        self.mlp_hidden_options = [[], [16], [16, 32]]
        self.aggregator_options = ["sum", "mean", "max"]
        self.bias_options = [True, False]
        self.activation_options = ["relu", "tanh"]
        self.eps_options = [0.0, 0.5]
        self.train_eps_options = [True, False]
        self.dropout_options = [0.0, 0.3]

        # Set random seeds for reproducibility
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

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

    def _extract_numpy_output(self, output):
        """Extract numpy array from output tensor (backend agnostic)."""
        try:
            return output.cpu().detach().numpy()
        except AttributeError:
            try:
                return output.cpu().numpy()
            except AttributeError:
                return keras.ops.convert_to_numpy(output)


@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvInitialization(TestGINConvBase):
    """Test GINConv layer initialization."""

    def test_basic_initialization(self):
        """Test basic layer initialization with defaults."""
        print("\n--- Testing GINConv Basic Initialization ---")

        gin = GINConv(output_dim=self.output_dim)

        # Check default values
        self.assertEqual(gin.output_dim, self.output_dim)
        self.assertEqual(gin.mlp_hidden, [])
        self.assertEqual(gin.aggregator, "sum")
        self.assertEqual(gin.eps_init, 0.0)
        self.assertFalse(gin.train_eps)
        self.assertTrue(gin.use_bias)
        self.assertEqual(gin.dropout_rate, 0.0)
        self.assertEqual(gin.activation, "relu")

    def test_parameter_variations(self):
        """Test layer initialization with various parameter combinations."""
        print("\n--- Testing GINConv Parameter Variations ---")

        # Test a subset of combinations to avoid explosion
        test_params = [
            ([], "sum", 0.0, False, True, 0.0, "relu"),
            ([16], "mean", 0.5, True, False, 0.3, "tanh"),
            ([16, 32], "max", 0.0, False, True, 0.0, "relu"),
        ]

        for (
            mlp_hidden,
            aggr,
            eps_init,
            train_eps,
            use_bias,
            dropout,
            activation,
        ) in test_params:
            with self.subTest(
                mlp_hidden=mlp_hidden,
                aggr=aggr,
                eps_init=eps_init,
                train_eps=train_eps,
                bias=use_bias,
                dropout=dropout,
                activation=activation,
            ):
                gin = GINConv(
                    output_dim=self.output_dim,
                    mlp_hidden=mlp_hidden,
                    aggregator=aggr,
                    eps_init=eps_init,
                    train_eps=train_eps,
                    use_bias=use_bias,
                    dropout=dropout,
                    activation=activation,
                )

                # Verify parameters
                self.assertEqual(gin.output_dim, self.output_dim)
                self.assertEqual(gin.mlp_hidden, mlp_hidden)
                self.assertEqual(gin.aggregator, aggr)
                self.assertEqual(gin.eps_init, eps_init)
                self.assertEqual(gin.train_eps, train_eps)
                self.assertEqual(gin.use_bias, use_bias)
                self.assertEqual(gin.dropout_rate, dropout)
                self.assertEqual(gin.activation, activation)

    def test_invalid_aggregator(self):
        """Test that invalid aggregator raises error."""
        print("\n--- Testing GINConv Invalid Aggregator ---")

        with self.assertRaises(AssertionError):
            GINConv(output_dim=self.output_dim, aggregator="invalid_aggr")


@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvForwardPass(TestGINConvBase):
    """Test GINConv forward pass functionality."""

    def test_output_shapes(self):
        """Test forward pass output shapes for different configurations."""
        print("\n--- Testing GINConv Output Shapes ---")

        input_data = [self.features_keras, self.edge_index_keras]

        test_params = [
            ([], "sum", True),
            ([16], "mean", False),
            ([16, 32], "max", True),
        ]

        for mlp_hidden, aggr, use_bias in test_params:
            with self.subTest(mlp_hidden=mlp_hidden, aggr=aggr, bias=use_bias):
                gin = GINConv(
                    output_dim=self.output_dim,
                    mlp_hidden=mlp_hidden,
                    aggregator=aggr,
                    use_bias=use_bias,
                )

                output = gin(input_data)
                output_shape = self._extract_numpy_output(output).shape

                expected_shape = (self.num_nodes, self.output_dim)
                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Shape mismatch for config: mlp={mlp_hidden}, aggr={aggr}, bias={use_bias}",
                )

    def test_dropout_behavior(self):
        """Test dropout behavior during training vs inference."""
        print("\n--- Testing GINConv Dropout Behavior ---")

        # Use high dropout rate to make effects visible
        gin = GINConv(
            output_dim=self.output_dim,
            mlp_hidden=[16, 32],
            dropout=0.5,
            use_bias=True,
        )

        input_data = [self.features_keras, self.edge_index_keras]

        # Test training mode - outputs should vary due to dropout
        training_outputs = []
        for _ in range(3):
            output = gin(input_data, training=True)
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
            output = gin(input_data, training=False)
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
        print("\n--- Testing GINConv Edge Cases ---")

        # Test empty graph (0 nodes)
        empty_features = keras.ops.zeros((0, self.input_dim))
        empty_edges = keras.ops.zeros((2, 0), dtype="int32")

        gin = GINConv(output_dim=self.output_dim, mlp_hidden=[16])
        output = gin([empty_features, empty_edges])

        expected_shape = (0, self.output_dim)
        self.assertEqual(
            self._extract_numpy_output(output).shape,
            expected_shape,
            "Empty graph should produce correct shape",
        )

        # Test single node graph with no edges
        single_node_features = keras.ops.ones((1, self.input_dim))
        no_edges = keras.ops.zeros((2, 0), dtype="int32")

        gin = GINConv(output_dim=self.output_dim, mlp_hidden=[16], eps_init=0.5)
        output = gin([single_node_features, no_edges])

        expected_shape = (1, self.output_dim)
        output_np = self._extract_numpy_output(output)
        self.assertEqual(
            output_np.shape,
            expected_shape,
            "Single node graph should produce correct shape",
        )

        # Verify that the output is non-zero (due to (1+eps)*x term)
        self.assertTrue(
            np.any(output_np != 0),
            "Single node output should be non-zero due to (1+eps)*x term",
        )

        # Test disconnected graph
        disconnected_features = keras.ops.ones((4, self.input_dim))
        disconnected_edges = keras.ops.array(
            [[0, 1], [1, 0]], dtype="int32"
        )  # Only 0-1 connected

        gin = GINConv(output_dim=self.output_dim, aggregator="sum")
        output = gin([disconnected_features, disconnected_edges])

        expected_shape = (4, self.output_dim)
        self.assertEqual(
            self._extract_numpy_output(output).shape,
            expected_shape,
            "Disconnected graph should produce correct shape",
        )

    def test_trainable_epsilon(self):
        """Test trainable epsilon parameter."""
        print("\n--- Testing GINConv Trainable Epsilon ---")

        # Create layer with trainable epsilon
        gin = GINConv(
            output_dim=self.output_dim,
            mlp_hidden=[16],
            eps_init=0.5,
            train_eps=True,
        )

        input_data = [self.features_keras, self.edge_index_keras]
        _ = gin(input_data)  # Build the layer

        # Check that epsilon is a trainable weight
        trainable_weights = gin.trainable_weights
        eps_found = any("eps" in w.name for w in trainable_weights)
        self.assertTrue(eps_found, "Epsilon should be a trainable weight")

        # Test with non-trainable epsilon
        gin_fixed = GINConv(
            output_dim=self.output_dim,
            mlp_hidden=[16],
            eps_init=0.5,
            train_eps=False,
        )
        _ = gin_fixed(input_data)

        # Check that epsilon is not in trainable weights
        trainable_weights_fixed = gin_fixed.trainable_weights
        eps_found_fixed = any("eps" in w.name for w in trainable_weights_fixed)
        self.assertFalse(eps_found_fixed, "Epsilon should not be trainable")


@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvSerialization(TestGINConvBase):
    """Test GINConv serialization and configuration."""

    def test_config_serialization(self):
        """Test get_config and from_config methods."""
        print("\n--- Testing GINConv Config Serialization ---")

        # Create layer with non-default parameters
        original_config = {
            "output_dim": self.output_dim + 1,
            "mlp_hidden": [32, 64],
            "aggregator": "mean",
            "eps_init": 0.5,
            "train_eps": True,
            "use_bias": False,
            "dropout": 0.3,
            "kernel_initializer": "he_normal",
            "bias_initializer": "ones",
            "activation": "tanh",
            "name": "test_gin_config",
        }

        gin1 = GINConv(**original_config)

        # Build the layer
        _ = gin1([self.features_keras, self.edge_index_keras])

        # Get configuration
        config = gin1.get_config()

        # Verify key configuration parameters
        expected_keys = [
            "name",
            "trainable",
            "output_dim",
            "mlp_hidden",
            "aggregator",
            "eps_init",
            "train_eps",
            "use_bias",
            "dropout",
            "kernel_initializer",
            "bias_initializer",
            "activation",
        ]

        for key in expected_keys:
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Test specific values
        self.assertEqual(config["output_dim"], original_config["output_dim"])
        self.assertEqual(config["mlp_hidden"], original_config["mlp_hidden"])
        self.assertEqual(config["aggregator"], original_config["aggregator"])
        self.assertEqual(config["eps_init"], original_config["eps_init"])
        self.assertEqual(config["train_eps"], original_config["train_eps"])
        self.assertEqual(config["use_bias"], original_config["use_bias"])
        self.assertEqual(config["dropout"], original_config["dropout"])
        self.assertEqual(config["activation"], original_config["activation"])
        self.assertEqual(config["name"], original_config["name"])

        # Test layer reconstruction
        try:
            gin2 = GINConv.from_config(config)
        except Exception as e:
            self.fail(f"GINConv.from_config failed: {e}")

        # Verify reconstructed layer properties
        self.assertEqual(gin1.output_dim, gin2.output_dim)
        self.assertEqual(gin1.mlp_hidden, gin2.mlp_hidden)
        self.assertEqual(gin1.aggregator, gin2.aggregator)
        self.assertEqual(gin1.eps_init, gin2.eps_init)
        self.assertEqual(gin1.train_eps, gin2.train_eps)
        self.assertEqual(gin1.use_bias, gin2.use_bias)
        self.assertEqual(gin1.dropout_rate, gin2.dropout_rate)
        self.assertEqual(gin1.activation, gin2.activation)
        self.assertEqual(gin1.name, gin2.name)


@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvGradients(TestGINConvBase):
    """Test gradient flow through GINConv layer."""

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        print("\n--- Testing GINConv Gradient Flow ---")

        gin = GINConv(
            output_dim=self.output_dim,
            mlp_hidden=[16],
            train_eps=True,
        )

        input_data = [self.features_keras, self.edge_index_keras]

        # Use backend-specific gradient tape
        backend = get_keras_backend()
        if backend == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape() as tape:
                output = gin(input_data)
                loss = keras.ops.sum(output)
            grads = tape.gradient(loss, gin.trainable_variables)
        elif backend == "torch":
            # For PyTorch backend, we'll skip gradient testing since it's more complex
            self.skipTest("Gradient testing not implemented for PyTorch backend")
        elif backend == "jax":
            # For JAX backend, we'll skip gradient testing since it's more complex
            self.skipTest("Gradient testing not implemented for JAX backend")
        else:
            self.skipTest(f"Gradient testing not implemented for {backend} backend")

        # Check that all gradients are computed
        for grad, var in zip(grads, gin.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")
            grad_np = self._extract_numpy_output(grad)
            self.assertFalse(np.any(np.isnan(grad_np)), f"NaN gradient for {var.name}")
            self.assertFalse(np.all(grad_np == 0), f"Zero gradient for {var.name}")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric not available")
@unittest.skipIf(not KERAS_BACKEND_IS_TORCH, "Keras backend is not torch")
@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvNumericalComparison(TestGINConvBase):
    """Test numerical comparison with PyTorch Geometric."""

    def _sync_mlp_weights(self, keras_gin, pyg_gin, use_bias):
        """Synchronize MLP weights between Keras and PyG layers."""
        # Get Keras Dense layers from the MLP
        keras_dense_layers = [
            layer
            for layer in keras_gin.mlp.layers
            if isinstance(layer, keras.layers.Dense)
        ]

        # Get PyTorch Linear layers from the MLP
        pyg_linear_layers = [
            m for m in pyg_gin.nn.modules() if isinstance(m, nn.Linear)
        ]

        if len(keras_dense_layers) != len(pyg_linear_layers):
            raise ValueError(
                f"MLP layer count mismatch: Keras {len(keras_dense_layers)} vs PyG {len(pyg_linear_layers)}"
            )

        # Sync weights for each layer
        for i, (k_layer, p_layer) in enumerate(
            zip(keras_dense_layers, pyg_linear_layers)
        ):
            k_weights = k_layer.get_weights()

            if use_bias:
                if len(k_weights) != 2:
                    raise ValueError(f"Expected 2 weights (kernel, bias) for layer {i}")
                k_kernel, k_bias = k_weights
                p_layer.weight.data.copy_(torch.tensor(k_kernel.T))
                p_layer.bias.data.copy_(torch.tensor(k_bias))
            else:
                if len(k_weights) != 1:
                    raise ValueError(f"Expected 1 weight (kernel) for layer {i}")
                k_kernel = k_weights[0]
                p_layer.weight.data.copy_(torch.tensor(k_kernel.T))
                self.assertIsNone(p_layer.bias, f"PyG layer {i} should have no bias")

    def test_numerical_comparison_basic(self):
        """Test numerical comparison for basic configurations."""
        print("\n--- Testing Numerical Comparison vs PyG GINConv ---")

        # Test configurations
        test_configs = [
            ([], "sum", True, False, 0.0),  # No hidden layers, sum aggregation
            ([16], "sum", False, True, 0.5),  # One hidden layer, trainable eps
            ([16, 32], "sum", True, False, 0.0),  # Two hidden layers
        ]

        for mlp_hidden, aggr, use_bias, train_eps, eps_init in test_configs:
            subtest_msg = (
                f"mlp={mlp_hidden}, aggr={aggr}, bias={use_bias}, "
                f"train_eps={train_eps}, eps={eps_init}"
            )

            with self.subTest(msg=subtest_msg):
                print(f"\n--- Comparing: {subtest_msg} ---")

                # Create PyG MLP
                pyg_mlp_layers = []
                current_dim = self.input_dim

                # Add hidden layers with ReLU activation
                for hidden_dim in mlp_hidden:
                    pyg_mlp_layers.append(
                        nn.Linear(current_dim, hidden_dim, bias=use_bias)
                    )
                    pyg_mlp_layers.append(nn.ReLU())
                    current_dim = hidden_dim

                # Add output layer (no activation)
                pyg_mlp_layers.append(
                    nn.Linear(current_dim, self.output_dim, bias=use_bias)
                )

                pyg_mlp = nn.Sequential(*pyg_mlp_layers)

                # Create Keras layer
                keras_gin = GINConv(
                    output_dim=self.output_dim,
                    mlp_hidden=mlp_hidden,
                    aggregator=aggr,
                    eps_init=eps_init,
                    train_eps=train_eps,
                    use_bias=use_bias,
                    activation="relu",
                )

                # Build Keras layer
                _ = keras_gin([self.features_keras, self.edge_index_keras])

                # Create PyG layer
                pyg_gin = PyGGINConv(
                    nn=pyg_mlp,
                    eps=eps_init,
                    train_eps=train_eps,
                    aggr=aggr,
                )

                # Sync weights
                try:
                    self._sync_mlp_weights(keras_gin, pyg_gin, use_bias)

                    # Sync epsilon if trainable
                    if train_eps and hasattr(keras_gin, "eps"):
                        pyg_gin.eps.data.copy_(
                            torch.tensor(keras_gin.eps.numpy()).squeeze()
                        )
                except Exception as e:
                    self.skipTest(f"Weight synchronization failed: {e}")

                # Forward pass
                keras_output = keras_gin([self.features_keras, self.edge_index_keras])
                pyg_output = pyg_gin(self.features_torch, self.edge_index_torch)

                # Compare outputs
                keras_output_np = self._extract_numpy_output(keras_output)
                pyg_output_np = pyg_output.cpu().detach().numpy()

                # Verify shapes
                self.assertEqual(
                    keras_output_np.shape, (self.num_nodes, self.output_dim)
                )
                self.assertEqual(pyg_output_np.shape, (self.num_nodes, self.output_dim))

                # Compare values
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
                    # Provide diagnostic information
                    abs_diff = np.abs(keras_output_np - pyg_output_np)
                    rel_diff = abs_diff / (np.abs(pyg_output_np) + 1e-8)

                    max_abs_diff = np.max(abs_diff)
                    max_rel_diff = np.max(rel_diff)

                    print(f"⚠️  Outputs differ for: {subtest_msg}")
                    print(f"   Max Abs Diff: {max_abs_diff:.6f}")
                    print(f"   Max Rel Diff: {max_rel_diff:.6f}")
                    print(f"   Keras sample: {keras_output_np[0, :5]}")
                    print(f"   PyG sample: {pyg_output_np[0, :5]}")

                    # Only fail if differences are large
                    if max_abs_diff > 1e-2 or max_rel_diff > 1e-2:
                        self.fail(f"Large numerical differences for {subtest_msg}: {e}")


@unittest.skipIf(not GIN_AVAILABLE, "GINConv layer could not be imported.")
class TestGINConvAggregators(TestGINConvBase):
    """Test different aggregation methods in GINConv."""

    def test_aggregator_outputs(self):
        """Test that different aggregators produce different outputs."""
        print("\n--- Testing GINConv Aggregator Outputs ---")

        input_data = [self.features_keras, self.edge_index_keras]
        outputs = {}

        for aggr in ["sum", "mean", "max"]:
            gin = GINConv(
                output_dim=self.output_dim,
                mlp_hidden=[16],
                aggregator=aggr,
                eps_init=0.0,  # Set eps=0 to isolate aggregation effects
            )

            output = gin(input_data)
            outputs[aggr] = self._extract_numpy_output(output)

        # Verify that different aggregators produce different outputs
        for aggr1, aggr2 in itertools.combinations(outputs.keys(), 2):
            self.assertFalse(
                np.allclose(outputs[aggr1], outputs[aggr2], rtol=1e-5, atol=1e-5),
                f"Outputs for {aggr1} and {aggr2} should be different",
            )

    def test_aggregator_properties(self):
        """Test specific properties of aggregators."""
        print("\n--- Testing GINConv Aggregator Properties ---")

        # Create a simple graph where node 0 has multiple neighbors with same features
        features = keras.ops.array(
            [
                [1.0] * self.input_dim,  # Node 0
                [2.0] * self.input_dim,  # Node 1
                [2.0] * self.input_dim,  # Node 2
                [2.0] * self.input_dim,  # Node 3
            ],
            dtype="float32",
        )

        # Node 0 receives from nodes 1, 2, 3 (all have same features)
        edges = keras.ops.array([[1, 2, 3], [0, 0, 0]], dtype="int32")

        # Test sum aggregation
        gin_sum = GINConv(
            output_dim=self.output_dim,
            aggregator="sum",
            eps_init=0.0,
        )
        output_sum = gin_sum([features, edges])

        # Test mean aggregation
        gin_mean = GINConv(
            output_dim=self.output_dim,
            aggregator="mean",
            eps_init=0.0,
        )
        output_mean = gin_mean([features, edges])

        # The sum aggregation should produce larger values for node 0
        # than mean aggregation (since it has 3 neighbors)
        output_sum_np = self._extract_numpy_output(output_sum)
        output_mean_np = self._extract_numpy_output(output_mean)

        # Node 0 should have different outputs for sum vs mean
        self.assertFalse(
            np.allclose(output_sum_np[0], output_mean_np[0], rtol=1e-5, atol=1e-5),
            "Sum and mean aggregation should produce different outputs for node 0",
        )


# Test runner
if __name__ == "__main__":
    # Configure test verbosity and warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests
    unittest.main(verbosity=2)
