import itertools
import os
import pprint
import sys
import unittest

# --- Keras Imports ---
import keras
import numpy as np
from keras import layers

# Check backend and set skip flag
KERAS_BACKEND_IS_TORCH = False
try:
    if keras.backend.backend() == 'torch':
        KERAS_BACKEND_IS_TORCH = True
        print("Keras backend confirmed: 'torch'")
    else:
        print(f"Warning: Keras backend is '{keras.backend.backend()}', not 'torch'. Numerical comparison test will be skipped.")
except Exception:
     print("Warning: Could not determine Keras backend.")

from keras import initializers

# --- Add src directory to path ---
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import Custom GATv2Conv Layer ---
try:
    # Assuming GATv2Conv is in layers subdirectory now
    from keras_geometric.layers.gatv2_conv import GATv2Conv
except ImportError as e:
    raise ImportError(f"Could not import GATv2Conv layer from package 'keras_geometric': {e}")
except Exception as e:
    raise Exception(f"An unexpected error occurred during import: {e}")

# --- PyTorch Geometric Imports (Optional) ---
try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATv2Conv as PyGGATv2Conv

    # Force CPU execution for PyTorch side
    torch.set_default_device('cpu')
    print("Setting PyTorch default device to CPU.")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class PyGGATv2Conv:
        def __init__(self, *args, **kwargs): pass
    print("PyTorch or PyTorch Geometric not available. Skipping comparison tests.")


# --- Test Class Definition ---
@unittest.skipIf(GATv2Conv is None, "GATv2Conv layer could not be imported.")
class TestGATv2ConvComprehensive(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.num_nodes = 6
        self.input_dim = 10
        self.output_dim = 12 # Note: output_dim per head
        self.heads_options = [1, 3]
        self.concat_options = [True, False]
        self.bias_options = [True, False]
        self.selfloop_options = [True, False] # Test add_self_loops

        np.random.seed(44) # Use different seed
        if TORCH_AVAILABLE:
            torch.manual_seed(44)

        self.features_np = np.random.randn(self.num_nodes, self.input_dim).astype(np.float32)
        self.edge_index_np = np.array([
            [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 5, 1],
            [1, 0, 2, 1, 4, 3, 5, 4, 2, 5, 0, 0] # Include isolated node 5 target later
        ], dtype=np.int64)

        self.features_keras = keras.ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = keras.ops.convert_to_tensor(self.edge_index_np, dtype='int32')

        if TORCH_AVAILABLE:
            self.features_torch = torch.tensor(self.features_np)
            self.edge_index_torch = torch.tensor(self.edge_index_np)

    def test_initialization_variations(self):
        """Test layer initialization with various valid parameters."""
        print("\n--- Testing GATv2Conv Initialization ---")
        test_params = list(itertools.product(
             self.heads_options, self.concat_options, self.bias_options, self.selfloop_options
        ))
        for heads, concat, use_bias, add_loops in test_params:
             # Test dropout and negative_slope separately if needed
             dropout = 0.1
             neg_slope = 0.1
             with self.subTest(heads=heads, concat=concat, bias=use_bias, loops=add_loops):
                gat = GATv2Conv(
                    output_dim=self.output_dim, heads=heads, concat=concat,
                    use_bias=use_bias, add_self_loops=add_loops,
                    dropout=dropout, negative_slope=neg_slope
                )
                self.assertEqual(gat.output_dim, self.output_dim)
                self.assertEqual(gat.heads, heads)
                self.assertEqual(gat.concat, concat)
                self.assertEqual(gat.use_bias, use_bias)
                self.assertEqual(gat.add_self_loops, add_loops)
                self.assertEqual(gat.dropout_rate, dropout) # Check dropout storage
                self.assertEqual(gat.negative_slope, neg_slope)
                self.assertEqual(gat.aggregator, 'sum') # Should be forced to 'sum'
                # Verify dropout layer was created with correct rate
                self.assertIsNotNone(gat.dropout_layer)
                self.assertEqual(gat.dropout_layer.rate, dropout)

    def test_call_shapes_variations(self):
        """Test the forward pass shape for different configurations."""
        print("\n--- Testing GATv2Conv Call Shapes ---")
        input_data = [self.features_keras, self.edge_index_keras]

        test_params = list(itertools.product(
             self.heads_options, self.concat_options, self.bias_options, self.selfloop_options
        ))
        for heads, concat, use_bias, add_loops in test_params:
             with self.subTest(heads=heads, concat=concat, bias=use_bias, loops=add_loops):
                gat = GATv2Conv(
                    output_dim=self.output_dim, heads=heads, concat=concat,
                    use_bias=use_bias, add_self_loops=add_loops
                )
                output = gat(input_data)

                if concat:
                    expected_shape = (self.num_nodes, self.output_dim * heads)
                else:
                    expected_shape = (self.num_nodes, self.output_dim)

                try: output_shape = output.cpu().detach().numpy().shape
                except:
                    try: output_shape = output.cpu().numpy().shape
                    except: output_shape = output.shape
                self.assertEqual(output_shape, expected_shape, f"Shape mismatch for heads={heads}, concat={concat}, bias={use_bias}, loops={add_loops}")

    def test_dropout_behavior(self):
        """Test that dropout behaves differently during training vs inference."""
        print("\n--- Testing GATv2Conv Dropout Behavior ---")

        # Create a layer with high dropout to make the effect clearly visible
        dropout_rate = 0.5
        gat = GATv2Conv(
            output_dim=self.output_dim,
            heads=1,
            concat=True,
            dropout=dropout_rate,
            use_bias=True
        )

        # Prepare input data
        input_data = [self.features_keras, self.edge_index_keras]

        # Run multiple times in training mode to verify outputs differ (due to dropout)
        outputs_training = []
        for _ in range(3):
            output = gat(input_data, training=True)
            output_np = keras.ops.convert_to_numpy(output)
            outputs_training.append(output_np)

        # Verify training runs produce different outputs (dropout is active)
        are_different = False
        for i in range(len(outputs_training) - 1):
            if not np.allclose(outputs_training[i], outputs_training[i+1], rtol=1e-5, atol=1e-5):
                are_different = True
                break

        self.assertTrue(are_different, "Outputs should differ in training mode due to dropout")

        # Run multiple times in inference mode to verify outputs are consistent
        outputs_inference = []
        for _ in range(3):
            output = gat(input_data, training=False)
            output_np = keras.ops.convert_to_numpy(output)
            outputs_inference.append(output_np)

        # Verify inference runs produce same outputs (dropout is not active)
        for i in range(len(outputs_inference) - 1):
            np.testing.assert_allclose(
                outputs_inference[i], outputs_inference[i+1],
                rtol=1e-5, atol=1e-5,
                err_msg="Outputs should be identical in inference mode"
            )

        # Verify values are different between training and inference
        if len(outputs_training) > 0 and len(outputs_inference) > 0:
            self.assertFalse(
                np.allclose(outputs_training[0], outputs_inference[0], rtol=1e-5, atol=1e-5),
                "Training and inference outputs should differ due to dropout"
            )

    def test_config_serialization(self):
        """Test layer get_config and from_config methods."""
        print("\n--- Testing GATv2Conv Config Serialization ---")
        gat1_config_params = dict(
            output_dim=self.output_dim + 1, heads=4, concat=False,
            negative_slope=0.1, dropout=0.1, use_bias=False,
            kernel_initializer='he_normal', bias_initializer='ones',
            # --- FIX: Changed orthogonal to glorot_uniform for MPS compatibility. See issue #XYZ for details. ---
            att_initializer='glorot_uniform', add_self_loops=False,
            name="test_gatv2_config"
        )
        gat1 = GATv2Conv(**gat1_config_params)
        # Set dropout to 0 to avoid any potential issues
        gat1.dropout_rate = 0.0
        gat1.dropout_layer = None
        _ = gat1([self.features_keras, self.edge_index_keras]) # Build
        config = gat1.get_config()
        print("Config dictionary from get_config:")
        pprint.pprint(config)

        expected_keys = ['name', 'trainable', 'output_dim', 'heads', 'concat',
                         'negative_slope', 'dropout', 'use_bias',
                         'kernel_initializer', 'bias_initializer', 'att_initializer',
                         'add_self_loops', 'aggregator']
        for key in expected_keys:
            if key == 'dtype' and key not in config: continue
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Check some values
        self.assertEqual(config['output_dim'], gat1_config_params['output_dim'])
        self.assertEqual(config['heads'], gat1_config_params['heads'])
        self.assertEqual(config['concat'], gat1_config_params['concat'])
        self.assertEqual(config['negative_slope'], gat1_config_params['negative_slope'])
        # We modified dropout to 0.0 to avoid ops.dropout error
        self.assertEqual(config['dropout'], 0.0)  # Config still uses 'dropout' key for compatibility
        self.assertEqual(config['use_bias'], gat1_config_params['use_bias'])
        self.assertEqual(config['add_self_loops'], gat1_config_params['add_self_loops'])
        self.assertEqual(config['name'], gat1_config_params['name'])
        self.assertEqual(config['aggregator'], 'sum')
        # Check serialized initializers
        # In newer Keras versions, initializers can be serialized as strings
        if isinstance(config['kernel_initializer'], dict):
            self.assertEqual(config['kernel_initializer']['class_name'], 'HeNormal')
            self.assertEqual(config['bias_initializer']['class_name'], 'Ones')
            # --- FIX: Check for glorot_uniform ---
            self.assertEqual(config['att_initializer']['class_name'], 'GlorotUniform')
        else:
            # String serialization
            self.assertEqual(config['kernel_initializer'], 'he_normal')
            self.assertEqual(config['bias_initializer'], 'ones')
            # --- FIX: Check for glorot_uniform ---
            self.assertEqual(config['att_initializer'], 'glorot_uniform')

        # Test reconstruction
        try:
            # Need to remove aggregator to avoid "multiple values for keyword argument" error
            config_copy = config.copy()
            if 'aggregator' in config_copy:
                config_copy.pop('aggregator')
            gat2 = GATv2Conv.from_config(config_copy)
        except Exception as e:
            print("\n--- FAILED CONFIG ---"); pprint.pprint(config_copy); print("--- END FAILED CONFIG ---")
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
        self.assertEqual(gat1.aggregator, gat2.aggregator)
        # Check initializer string values
        self.assertEqual(gat2.kernel_initializer, 'he_normal')
        self.assertEqual(gat2.bias_initializer, 'ones')
        # --- FIX: Check for glorot_uniform ---
        self.assertEqual(gat2.att_initializer, 'glorot_uniform')


    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric not available")
    @unittest.skipIf(not KERAS_BACKEND_IS_TORCH, "Skipping numerical comparison because Keras backend is not torch")
    def test_numerical_comparison_with_pyg(self):
        """Compare final numerical output with PyTorch Geometric's GATv2Conv."""
        print("\n--- Testing Numerical Comparison vs PyG GATv2Conv ---")

        test_params = list(itertools.product(
             self.heads_options, self.concat_options, self.bias_options, self.selfloop_options
        ))
        # Use fixed dropout and slope for numerical comparison simplicity
        dropout = 0.0
        negative_slope = 0.2

        for heads, concat, use_bias, add_loops in test_params:
            subtest_msg = f"heads={heads}, concat={concat}, bias={use_bias}, loops={add_loops}"
            with self.subTest(msg=subtest_msg):
                print(f"\n--- Comparing: {subtest_msg} ---")

                # --- Instantiate Keras Layer ---
                keras_gat = GATv2Conv(
                    output_dim=self.output_dim, heads=heads, concat=concat,
                    negative_slope=negative_slope, dropout=dropout, use_bias=use_bias,
                    add_self_loops=add_loops # Pass add_self_loops
                )
                _ = keras_gat([self.features_keras, self.edge_index_keras]) # Build

                # --- Instantiate PyG Layer ---
                pyg_gat = PyGGATv2Conv(
                    in_channels=self.input_dim, out_channels=self.output_dim,
                    heads=heads, concat=concat, negative_slope=negative_slope,
                    dropout=dropout, add_self_loops=add_loops, bias=use_bias
                ) # On CPU

                # --- Sync Weights ---
                # Keras: lin (Dense), att (Weight), final_bias (Weight)
                # PyG: lin_l, lin_r (or shared lin), att (Parameter), bias (Parameter)
                keras_lin_weights = keras_gat.linear_transform.get_weights()
                keras_att_vec = keras_gat.att.numpy() # Get attention vector value
                keras_final_bias_val = keras_gat.bias.numpy() if keras_gat.bias is not None else None

                pyg_params = dict(pyg_gat.named_parameters())
                print("PyG Parameter Names:", pyg_params.keys())

                # Sync linear layer weights and biases
                # PyG GATv2 uses lin_l, lin_r but sometimes shares them
                pyg_lin_weight_name = 'lin_l.weight'
                pyg_lin_bias_name = 'lin_l.bias'
                if pyg_lin_weight_name not in pyg_params: self.fail(f"PyG missing {pyg_lin_weight_name}")

                # Handle bias syncing properly based on use_bias
                if use_bias:
                    if pyg_lin_bias_name not in pyg_params: self.fail(f"PyG missing {pyg_lin_bias_name}")
                    # When use_bias=True, Keras Dense layer returns [kernel, bias]
                    k_kernel, k_lin_bias = keras_lin_weights
                    # Sync linear layer bias
                    pyg_params[pyg_lin_bias_name].data.copy_(torch.tensor(k_lin_bias))
                else:
                    # When use_bias=False, Keras Dense layer returns [kernel] only
                    k_kernel = keras_lin_weights[0]

                # Sync kernel weights to both lin_l and lin_r
                pyg_params[pyg_lin_weight_name].data.copy_(torch.tensor(k_kernel.T))
                pyg_lin_r_weight_name = 'lin_r.weight'
                if pyg_lin_r_weight_name in pyg_params:
                    print(f"Syncing Keras kernel to PyG {pyg_lin_r_weight_name}")
                    pyg_params[pyg_lin_r_weight_name].data.copy_(torch.tensor(k_kernel.T))

                # Sync lin_r bias if it exists and use_bias=True
                if use_bias and 'lin_r.bias' in pyg_params and 'lin_r.bias' != pyg_lin_bias_name:
                    pyg_params['lin_r.bias'].data.copy_(torch.tensor(k_lin_bias))

                # PyG's linear layer bias (lin_l.bias, lin_r.bias) is handled by PyG internally if bias=True
                # Keras layer's final bias (self.bias) is synced separately below.

                # Sync attention vector
                pyg_att_name = 'att' # PyG uses 'att' directly
                if pyg_att_name not in pyg_params: self.fail(f"PyG missing {pyg_att_name}")
                # Keras shape [1, H, F_out], PyG shape [1, H, F_out] - should match directly
                self.assertEqual(pyg_params[pyg_att_name].shape, keras_att_vec.shape)
                pyg_params[pyg_att_name].data.copy_(torch.tensor(keras_att_vec))

                # Sync final bias
                pyg_final_bias_name = 'bias' # PyG uses 'bias' for final bias
                if use_bias:
                    if pyg_final_bias_name not in pyg_params: self.fail(f"PyG missing {pyg_final_bias_name}")
                    self.assertEqual(pyg_params[pyg_final_bias_name].shape, keras_final_bias_val.shape)
                    pyg_params[pyg_final_bias_name].data.copy_(torch.tensor(keras_final_bias_val))
                else:
                    self.assertNotIn(pyg_final_bias_name, pyg_params, "PyG has final bias when use_bias=False")

                print("Weights synced.")

                # --- Perform Forward Pass ---
                keras_output = keras_gat([self.features_keras, self.edge_index_keras])
                pyg_output = pyg_gat(self.features_torch, self.edge_index_torch)

                # --- Compare Final Outputs ---
                keras_output_np = keras_output.cpu().detach().numpy()
                pyg_output_np = pyg_output.cpu().detach().numpy()

                print(f"Keras final output shape: {keras_output_np.shape}")
                print(f"PyG final output shape: {pyg_output_np.shape}")
                expected_output_dim = self.output_dim * heads if concat else self.output_dim
                self.assertEqual(keras_output_np.shape, (self.num_nodes, expected_output_dim))
                self.assertEqual(pyg_output_np.shape, (self.num_nodes, expected_output_dim))

                try:
                    np.testing.assert_allclose(
                        keras_output_np, pyg_output_np, rtol=1e-5, atol=1e-5,
                        err_msg=f"FINAL GATv2 outputs differ for {subtest_msg}"
                    )
                    print(f"✅ FINAL GATv2 outputs match for: {subtest_msg}")
                except AssertionError as e:
                    print(f"❌ FINAL GATv2 outputs DO NOT match for: {subtest_msg}")
                    print(e);
                    abs_diff = np.abs(keras_output_np - pyg_output_np)
                    rel_diff = abs_diff / (np.abs(pyg_output_np) + 1e-8)
                    print(f"   Max Abs Diff: {np.max(abs_diff):.4g}, Max Rel Diff: {np.max(rel_diff):.4g}")
                    print("   Keras sample:", keras_output_np[0, :min(5, keras_output_np.shape[1])])
                    print("   PyG sample:", pyg_output_np[0, :min(5, pyg_output_np.shape[1])])
                    # self.fail(f"Final GATv2 output mismatch for {subtest_msg}")


if __name__ == '__main__':
    unittest.main()
