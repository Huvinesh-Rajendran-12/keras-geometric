import unittest
import numpy as np
import os
import sys
import itertools # For parameterizing tests
import pprint

# --- Keras/TensorFlow Imports ---
import keras
# Check if the backend is set to 'torch'
# Allow skipping if backend is not torch, instead of hard assertion error
KERAS_BACKEND_IS_TORCH = False
try:
    if keras.backend.backend() == 'torch':
        KERAS_BACKEND_IS_TORCH = True
        print(f"Keras backend confirmed: 'torch'")
    else:
        print(f"Warning: Keras backend is '{keras.backend.backend()}', not 'torch'. Numerical comparison tests might fail.")
except Exception: # Handle cases where backend check might fail
     print("Warning: Could not determine Keras backend.")

from keras import Sequential, layers, initializers

# --- Add src directory to path ---
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import Custom GINConv Layer ---
try:
    # Assuming the user has fixed the issues in their gin_conv.py
    from keras_geometric.gin_conv import GINConv
    from keras_geometric.message_passing import MessagePassing # Base class might be needed
except ImportError as e:
    print(f"Could not import from package 'keras_geometric': {e}")
    GINConv = None
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    GINConv = None

# --- PyTorch Geometric Imports (Optional) ---
try:
    import torch
    import torch.nn as nn
    import torch_scatter
    from torch_geometric.nn import GINConv as PyGGINConv
    # Force CPU execution for PyTorch side for consistent comparison
    torch.set_default_device('cpu')
    print("Setting PyTorch default device to CPU.")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define placeholder classes if PyTorch/PyG are not available
    class PyGGINConv:
        def __init__(self, nn, **kwargs): pass
    class nn:
        Module = object
        Sequential = object
        Linear = object
        ReLU = object
        Tanh = object # Add Tanh if testing different activations
    class torch_scatter:
        @staticmethod
        def scatter(*args, **kwargs): pass
    print("PyTorch, PyTorch Geometric, or torch-scatter not available. Skipping comparison tests.")


# --- Test Class Definition ---
@unittest.skipIf(GINConv is None, "Custom GINConv layer could not be imported.")
class TestGINConvComprehensive(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.num_nodes = 7 # Use a slightly different number of nodes
        self.input_dim = 10
        self.output_dim = 15
        self.mlp_hidden_options = [
            [],             # Test direct linear transformation
            [16],           # Test one hidden layer
            [16, 20]        # Test two hidden layers
        ]
        self.aggregation_options = ['mean', 'max', 'sum'] # Use 'sum' for Keras side
        self.bias_options = [True, False]
        self.activation_options = ['relu', 'tanh'] # Test different activations

        # Use fixed seed for reproducibility
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

        # Generate common features and edge index
        self.features_np = np.random.randn(self.num_nodes, self.input_dim).astype(np.float32)
        # Generate a slightly more complex edge index
        self.edge_index_np = np.array([
            [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 6, 5, 1],  # Source nodes
            [1, 0, 2, 1, 4, 3, 5, 4, 2, 5, 5, 6, 6]   # Target nodes (ensure indices up to N-1 exist)
        ], dtype=np.int64)

        # Ensure Keras tensors respect torch default device if backend is torch
        self.features_keras = keras.ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = keras.ops.convert_to_tensor(self.edge_index_np, dtype='int32')

        if TORCH_AVAILABLE:
            # These will be on CPU due to torch.set_default_device
            self.features_torch = torch.tensor(self.features_np)
            self.edge_index_torch = torch.tensor(self.edge_index_np)

    def test_initialization_variations(self):
        """Test layer initialization with various valid parameters."""
        print("\n--- Testing Initialization Variations ---")
        for mlp_hidden, aggr, use_bias, activation in itertools.product(
            self.mlp_hidden_options, self.aggregation_options, self.bias_options, self.activation_options
        ):
            with self.subTest(mlp_hidden=mlp_hidden, aggr=aggr, use_bias=use_bias, activation=activation):
                gin = GINConv(
                    output_dim=self.output_dim,
                    mlp_hidden=mlp_hidden,
                    aggr=aggr,
                    use_bias=use_bias,
                    activation=activation
                )
                self.assertEqual(gin.output_dim, self.output_dim)
                self.assertEqual(gin.mlp_hidden, mlp_hidden)
                self.assertEqual(gin.aggr, aggr) # Assumes GINConv.__init__ passes aggr to super()
                self.assertEqual(gin.use_bias, use_bias)
                self.assertEqual(gin.activation, activation) # Check stored identifier

        # Test invalid aggregation
        with self.assertRaises(AssertionError, msg="AssertionError not raised for invalid aggregation"):
            GINConv(output_dim=self.output_dim, mlp_hidden=[16], aggr='invalid_aggr')

    def test_call_shapes_variations(self):
        """Test the forward pass shape for different configurations."""
        print("\n--- Testing Call Shapes Variations ---")
        input_data = [self.features_keras, self.edge_index_keras]
        expected_shape = (self.num_nodes, self.output_dim)

        for mlp_hidden, aggr, use_bias, activation in itertools.product(
             self.mlp_hidden_options, self.aggregation_options, self.bias_options, self.activation_options
        ):
             with self.subTest(mlp_hidden=mlp_hidden, aggr=aggr, use_bias=use_bias, activation=activation):
                gin = GINConv(
                    output_dim=self.output_dim,
                    mlp_hidden=mlp_hidden,
                    aggr=aggr,
                    use_bias=use_bias,
                    activation=activation
                )
                output = gin(input_data)
                # Use try-except for robust shape checking across backends/devices
                try:
                    output_shape = output.cpu().detach().numpy().shape
                except:
                    try: output_shape = output.cpu().numpy().shape
                    except:
                        try: output_shape = np.array(output).shape
                        except: output_shape = output.shape
                self.assertEqual(output_shape, expected_shape, f"Shape mismatch for config: {mlp_hidden}, {aggr}, {use_bias}, {activation}")

    def test_config_serialization(self):
        """Test layer get_config and from_config methods."""
        print("\n--- Testing Config Serialization ---")
        # Test with non-default values
        gin1 = GINConv(
            output_dim=self.output_dim + 1, # Different output dim
            mlp_hidden=[32, 64],
            aggr='max',
            use_bias=False,
            activation='tanh',
            kernel_initializer='he_normal',
            bias_initializer='ones',
            name="test_gin_config"
        )
        # Build layer to ensure all weights/attributes are created if needed by get_config
        _ = gin1([self.features_keras, self.edge_index_keras])
        config = gin1.get_config()

        pprint.pprint(config)

        # Check all expected keys are present
        expected_keys = ['name', 'trainable', 'dtype', 'output_dim', 'mlp_hidden',
                         'aggr', 'use_bias', 'kernel_initializer', 'bias_initializer',
                         'activation']
        for key in expected_keys:
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Check some values (assuming GINConv stores identifiers correctly now)
        self.assertEqual(config['output_dim'], self.output_dim + 1)
        self.assertEqual(config['mlp_hidden'], [32, 64])
        self.assertEqual(config['aggr'], 'max')
        self.assertFalse(config['use_bias'])
        self.assertEqual(config['activation'], 'tanh')
        self.assertEqual(config['kernel_initializer'], 'he_normal')
        self.assertEqual(config['bias_initializer'], 'ones')
        self.assertEqual(config['name'], "test_gin_config")

        # Test reconstruction
        try:
            gin2 = GINConv.from_config(config)
        except Exception as e:
            self.fail(f"GINConv.from_config failed: {e}\nConfig was: {config}")

        # Verify reconstructed layer properties
        self.assertEqual(gin1.output_dim, gin2.output_dim)
        self.assertEqual(gin1.mlp_hidden, gin2.mlp_hidden)
        self.assertEqual(gin1.aggr, gin2.aggr)
        self.assertEqual(gin1.use_bias, gin2.use_bias)
        # Compare stored identifiers
        self.assertEqual(gin1.activation, gin2.activation)
        self.assertEqual(gin1.kernel_initializer, gin2.kernel_initializer)
        self.assertEqual(gin1.bias_initializer, gin2.bias_initializer)
        self.assertEqual(gin1.name, gin2.name)


    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric or torch-scatter not available")
    @unittest.skipIf(not KERAS_BACKEND_IS_TORCH, "Skipping numerical comparison because Keras backend is not torch")
    def test_numerical_comparison_with_pyg(self):
        """Compare final numerical output with PyTorch Geometric's GINConv."""
        print("\n--- Testing Numerical Comparison vs PyG ---")

        # Combine options to test various configurations
        test_params = list(itertools.product(
            self.mlp_hidden_options, self.aggregation_options, self.bias_options, self.activation_options[:1] # Limit activations for brevity, focus on relu
        ))

        aggregation_map = {'mean': 'mean', 'max': 'max', 'sum': 'sum'} # Keras_aggr -> PyG_reduce

        for mlp_hidden, keras_aggr, use_bias, activation in test_params:
            pyg_reduce = aggregation_map[keras_aggr]
            subtest_msg = f"mlp={mlp_hidden}, aggr={keras_aggr}, bias={use_bias}, act={activation}"
            with self.subTest(msg=subtest_msg):
                print(f"\n--- Comparing: {subtest_msg} ---")

                 # --- Define PyG MLP structure (matching Keras) ---
                pyg_mlp_layers = []
                # Select PyG activation based on Keras identifier
                if activation == 'relu': pyg_activation_fn = nn.ReLU()
                elif activation == 'tanh': pyg_activation_fn = nn.Tanh()
                else: pyg_activation_fn = nn.Identity() # Or handle error

                current_dim = self.input_dim
                for i, hidden_units in enumerate(mlp_hidden):
                    pyg_mlp_layers.append(nn.Linear(current_dim, hidden_units, bias=use_bias))
                    pyg_mlp_layers.append(pyg_activation_fn) # Apply activation after hidden layers
                    current_dim = hidden_units
                # Final layer - NO activation
                pyg_mlp_layers.append(nn.Linear(current_dim, self.output_dim, bias=use_bias))
                pyg_mlp = nn.Sequential(*pyg_mlp_layers) # On CPU by default

                # --- Instantiate Keras Layer ---
                # Assuming user fixed GINConv: aggr passed to super, no activation on final MLP layer
                keras_gin = GINConv(
                    output_dim=self.output_dim, mlp_hidden=mlp_hidden,
                    aggr=keras_aggr, use_bias=use_bias, activation=activation
                )
                # Build layer
                _ = keras_gin([self.features_keras, self.edge_index_keras])

                # --- Instantiate PyG Layer ---
                pyg_gin = PyGGINConv(nn=pyg_mlp, aggr=pyg_reduce, train_eps=False) # On CPU

                # --- Sync Weights ---
                keras_dense_layers = [l for l in keras_gin.mlp.layers if isinstance(l, layers.Dense)]
                pyg_linear_layers = [m for m in pyg_gin.nn if isinstance(m, nn.Linear)]
                if len(keras_dense_layers) != len(pyg_linear_layers):
                    self.fail(f"MLP layer count mismatch Keras({len(keras_dense_layers)}) vs PyG({len(pyg_linear_layers)}) for {subtest_msg}")

                print(f"Syncing weights for {len(keras_dense_layers)} MLP layer(s)...")
                for i in range(len(keras_dense_layers)):
                    k_layer, p_layer = keras_dense_layers[i], pyg_linear_layers[i]
                    k_weights = k_layer.get_weights()
                    p_layer.to('cpu') # Ensure layer is on CPU before copy

                    if use_bias:
                        if len(k_weights) != 2: self.fail(f"Expected 2 weights (kernel, bias) but got {len(k_weights)} for layer {i} with use_bias=True")
                        k_kernel, k_bias = k_weights[0], k_weights[1]
                        p_layer.weight.data.copy_(torch.tensor(k_kernel.T))
                        p_layer.bias.data.copy_(torch.tensor(k_bias))
                        # Verification
                        synced_kernel_t = p_layer.weight.data.cpu().numpy().T
                        synced_bias = p_layer.bias.data.cpu().numpy()
                        np.testing.assert_allclose(k_kernel, synced_kernel_t, rtol=0, atol=0, err_msg=f"Kernel sync failed layer {i}")
                        np.testing.assert_allclose(k_bias, synced_bias, rtol=0, atol=0, err_msg=f"Bias sync failed layer {i}")
                    else: # No bias
                        if len(k_weights) != 1: self.fail(f"Expected 1 weight (kernel) but got {len(k_weights)} for layer {i} with use_bias=False")
                        k_kernel = k_weights[0]
                        # Check if PyG layer correctly has no bias
                        self.assertIsNone(p_layer.bias, f"PyG layer {i} has bias when use_bias=False")
                        p_layer.weight.data.copy_(torch.tensor(k_kernel.T))
                        # Verification
                        synced_kernel_t = p_layer.weight.data.cpu().numpy().T
                        np.testing.assert_allclose(k_kernel, synced_kernel_t, rtol=0, atol=0, err_msg=f"Kernel sync failed layer {i} (no bias)")

                print("Weights synced and verified.")

                # --- Perform Forward Pass ---
                # Keras GIN (uses torch backend on CPU)
                keras_output = keras_gin([self.features_keras, self.edge_index_keras])
                # PyG GIN (on CPU)
                pyg_output = pyg_gin(self.features_torch, self.edge_index_torch)

                # --- Compare Final Outputs ---
                keras_output_np = keras_output.cpu().detach().numpy()
                pyg_output_np = pyg_output.cpu().detach().numpy()

                print(f"Keras final output shape: {keras_output_np.shape}")
                print(f"PyG final output shape: {pyg_output_np.shape}")
                self.assertEqual(keras_output_np.shape, (self.num_nodes, self.output_dim))
                self.assertEqual(pyg_output_np.shape, (self.num_nodes, self.output_dim))

                try:
                    np.testing.assert_allclose(
                        keras_output_np, pyg_output_np, rtol=1e-5, atol=1e-5, # Standard tolerance
                        err_msg=f"FINAL outputs differ for {subtest_msg}"
                    )
                    print(f"✅ FINAL outputs match for: {subtest_msg}")
                except AssertionError as e:
                    print(f"❌ FINAL outputs DO NOT match for: {subtest_msg}")
                    print(e);
                    # Provide more context on failure
                    abs_diff = np.abs(keras_output_np - pyg_output_np)
                    rel_diff = abs_diff / (np.abs(pyg_output_np) + 1e-8) # Avoid division by zero
                    print(f"   Max Abs Diff: {np.max(abs_diff):.4g}, Max Rel Diff: {np.max(rel_diff):.4g}")
                    print("   Keras sample:", keras_output_np[0, :5])
                    print("   PyG sample:", pyg_output_np[0, :5])
                    # Optionally fail the test immediately on first mismatch
                    # self.fail(f"Final output mismatch for {subtest_msg}")


if __name__ == '__main__':
    unittest.main()
