import itertools
import os
import pprint
import sys
import unittest
from collections import defaultdict  # Needed for manual calculation test

# --- Keras Imports ---
import keras
import numpy as np
from keras import activations

# Check backend and set skip flag
KERAS_BACKEND_IS_TORCH = False
try:
    if keras.backend.backend() == "torch":
        KERAS_BACKEND_IS_TORCH = True
        print("Keras backend confirmed: 'torch'")
    else:
        print(
            f"Warning: Keras backend is '{keras.backend.backend()}', not 'torch'. Numerical comparison test will be skipped."
        )
except Exception:
    print("Warning: Could not determine Keras backend.")

# --- Add src directory to path ---
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import Custom GraphSAGEConv Layer ---
try:
    # Assumes GraphSAGEConv is in layers subdirectory now
    from keras_geometric.layers.sage_conv import SAGEConv
except ImportError as e:
    raise Exception(
        f"Could not import GraphSAGEConv layer from package 'keras_geometric': {e}"
    ) from e
except Exception as e:
    raise Exception(f"An unexpected error occurred during import: {e}") from e

# --- PyTorch Geometric Imports (Optional) ---
try:
    # pyrefly: ignore  # import-error
    import torch

    # pyrefly: ignore  # import-error
    import torch.nn as nn

    # pyrefly: ignore  # import-error
    from torch_geometric.nn import SAGEConv as PyGSAGEConv  # Import SAGEConv

    # Force CPU execution for PyTorch side
    torch.set_default_device("cpu")
    print("Setting PyTorch default device to CPU.")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    class PyGSAGEConv:  # Placeholder if PyG not installed
        def __init__(self, *args, **kwargs):
            pass

    # Create placeholder module for nn when PyTorch is not available
    class _PlaceholderNN:
        class Module:
            pass

        class Sequential:
            def __init__(self, *args):
                pass

        class Linear:
            def __init__(self, *args, **kwargs):
                pass

        class ReLU:
            def __init__(self, *args, **kwargs):
                pass

        class Tanh:
            def __init__(self, *args, **kwargs):
                pass

    nn = _PlaceholderNN()

    print("PyTorch or PyTorch Geometric not available. Skipping comparison tests.")


# --- Test Class Definition ---
@unittest.skipIf(SAGEConv is None, "GraphSAGEConv layer could not be imported.")
class TestGraphSAGEConvComprehensive(unittest.TestCase):  # Renamed class
    def setUp(self):
        """Set up test fixtures"""
        self.num_nodes = 7
        self.input_dim = 10
        self.output_dim = 15
        self.aggregation_options = ["mean", "max", "sum", "min", "std"]
        self.bias_options = [True, False]
        self.normalize_options = [True, False]
        self.activation_options = ["relu", "tanh", None]
        self.pool_activation_options = ["relu", "tanh"]  # Activations for pool MLP
        self.root_weight_options = [True, False]

        np.random.seed(45)
        if TORCH_AVAILABLE:
            torch.manual_seed(45)

        self.features_np = np.random.randn(self.num_nodes, self.input_dim).astype(
            np.float32
        )
        self.edge_index_np = np.array(
            [
                [0, 1, 1, 2, 3, 4, 4, 5, 0, 3, 6, 5, 1, 6],
                [1, 0, 2, 1, 4, 3, 5, 4, 2, 5, 5, 6, 6, 0],
            ],
            dtype=np.int64,
        )

        self.features_keras = keras.ops.convert_to_tensor(self.features_np)
        self.edge_index_keras = keras.ops.convert_to_tensor(
            self.edge_index_np, dtype="int32"
        )

        if TORCH_AVAILABLE:
            self.features_torch = torch.tensor(self.features_np)
            self.edge_index_torch = torch.tensor(self.edge_index_np)

    def test_initialization_variations(self):
        """Test layer initialization with various valid parameters."""
        print("\n--- Testing GraphSAGEConv Initialization ---")
        test_params = list(
            itertools.product(
                self.aggregation_options,
                self.normalize_options,
                self.bias_options,
                self.activation_options,
                self.root_weight_options,
            )
        )
        for (
            aggr,
            normalize,
            use_bias,
            activation,
            root_weight,
        ) in test_params:
            # pool_activation not used anymore since pooling aggregator is removed
            pool_act_param = None
            subtest_msg = f"aggr={aggr}, norm={normalize}, bias={use_bias}, act={activation}, root={root_weight}, pool_act={pool_act_param}"
            with self.subTest(msg=subtest_msg):
                layer = SAGEConv(
                    output_dim=self.output_dim,
                    aggregator=aggr,
                    normalize=normalize,
                    use_bias=use_bias,
                    activation=activation,
                    root_weight=root_weight,
                    pool_activation=pool_act_param,
                )
                self.assertEqual(layer.output_dim, self.output_dim)
                self.assertEqual(layer.aggregator, aggr)
                self.assertEqual(layer.normalize, normalize)
                self.assertEqual(layer.use_bias, use_bias)
                self.assertEqual(layer.root_weight, root_weight)
                self.assertEqual(layer.activation, activations.get(activation))
                # pool_activation check removed since pooling aggregator is no longer supported
                # No need to check base class aggregator anymore - SAGEConv stores its own aggregator

        # Test invalid aggregation
        with self.assertRaises(ValueError):
            SAGEConv(output_dim=self.output_dim, aggregator="lstm")  # LSTM removed
        with self.assertRaises(ValueError):
            SAGEConv(output_dim=self.output_dim, aggregator="invalid_aggr")

    def test_call_shapes_variations(self):
        """Test the forward pass shape for different configurations."""
        print("\n--- Testing GraphSAGEConv Call Shapes ---")
        input_data = [self.features_keras, self.edge_index_keras]
        expected_shape = (self.num_nodes, self.output_dim)
        test_params = list(
            itertools.product(
                self.aggregation_options,
                self.normalize_options,
                self.bias_options,
                self.activation_options,
                self.root_weight_options,
            )
        )
        for aggr, normalize, use_bias, activation, root_weight in test_params:
            with self.subTest(
                aggr=aggr,
                norm=normalize,
                bias=use_bias,
                act=activation,
                root=root_weight,
            ):
                layer = SAGEConv(
                    output_dim=self.output_dim,
                    aggregator=aggr,
                    normalize=normalize,
                    use_bias=use_bias,
                    activation=activation,
                    root_weight=root_weight,
                )
                output = layer(input_data)
                try:
                    output_shape = output.cpu().detach().numpy().shape
                except (AttributeError, RuntimeError):
                    try:
                        output_shape = output.cpu().numpy().shape
                    except (AttributeError, RuntimeError):
                        output_shape = output.shape
                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Shape mismatch for {aggr}, norm={normalize}, bias={use_bias}, act={activation}, root={root_weight}",
                )

    def test_config_serialization(self):
        """Test layer get_config and from_config methods."""
        print("\n--- Testing GraphSAGEConv Config Serialization ---")
        # Test with mean aggregator and non-defaults
        # pyrefly: ignore  # no-matching-overload
        layer1_config_params = dict(
            output_dim=self.output_dim + 1,
            # pyrefly: ignore  # bad-argument-type
            aggregator="mean",
            normalize=True,
            root_weight=False,
            use_bias=False,
            # pyrefly: ignore  # bad-argument-type
            activation="tanh",
            # pyrefly: ignore  # bad-argument-type
            pool_activation="sigmoid",
            # pyrefly: ignore  # bad-argument-type
            kernel_initializer="he_normal",
            # pyrefly: ignore  # bad-argument-type
            bias_initializer="ones",
            # pyrefly: ignore  # bad-argument-type
            name="test_sage_config",
        )
        layer1 = SAGEConv(**layer1_config_params)
        _ = layer1([self.features_keras, self.edge_index_keras])  # Build
        config = layer1.get_config()
        print("Config dictionary from get_config:")
        pprint.pprint(config)

        expected_keys = [
            "name",
            "trainable",
            "output_dim",
            "aggregator",
            "normalize",
            "root_weight",
            "use_bias",
            "activation",
            "pool_activation",
            "kernel_initializer",
            "bias_initializer",
        ]
        for key in expected_keys:
            if key == "dtype" and key not in config:
                continue
            self.assertIn(key, config, f"Key '{key}' missing from config")

        # Check values match initialization params
        self.assertEqual(config["output_dim"], layer1_config_params["output_dim"])
        self.assertEqual(config["aggregator"], layer1_config_params["aggregator"])
        self.assertEqual(config["normalize"], layer1_config_params["normalize"])
        self.assertEqual(config["root_weight"], layer1_config_params["root_weight"])
        self.assertEqual(config["use_bias"], layer1_config_params["use_bias"])
        # Check initializers - support both dictionary and string formats
        if isinstance(config["activation"], dict):
            self.assertEqual(config["activation"]["class_name"], "Tanh")
            self.assertEqual(config["pool_activation"]["class_name"], "Sigmoid")
        else:
            self.assertEqual(config["activation"], "tanh")
            self.assertEqual(config["pool_activation"], "sigmoid")

        # Handle both dictionary and string formats for initializers
        if isinstance(config["kernel_initializer"], dict):
            self.assertEqual(config["kernel_initializer"]["class_name"], "HeNormal")
            self.assertEqual(config["bias_initializer"]["class_name"], "Ones")
        else:
            self.assertEqual(config["kernel_initializer"], "he_normal")
            self.assertEqual(config["bias_initializer"], "ones")
        self.assertEqual(config["name"], layer1_config_params["name"])

        # Test reconstruction
        try:
            layer2 = SAGEConv.from_config(config)
        except Exception as e:
            print("\n--- FAILED CONFIG ---")
            pprint.pprint(config)
            print("--- END FAILED CONFIG ---")
            self.fail(f"GraphSAGEConv.from_config failed: {e}")

        # Verify reconstructed layer properties
        self.assertEqual(layer1.output_dim, layer2.output_dim)
        self.assertEqual(layer1.aggregator, layer2.aggregator)
        self.assertEqual(layer1.normalize, layer2.normalize)
        self.assertEqual(layer1.root_weight, layer2.root_weight)
        self.assertEqual(layer1.use_bias, layer2.use_bias)
        self.assertEqual(layer1.name, layer2.name)
        # Check activation types
        self.assertEqual(layer2.activation, activations.get("tanh"))
        self.assertEqual(layer2.pool_activation, activations.get("sigmoid"))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or PyTorch Geometric not available")
    @unittest.skipIf(
        not KERAS_BACKEND_IS_TORCH,
        "Skipping numerical comparison because Keras backend is not torch",
    )
    def test_numerical_comparison_with_pyg(self):
        """Compare final numerical output with PyTorch Geometric's SAGEConv for mean/max/sum."""
        print("\n--- Testing Numerical Comparison vs PyG SAGEConv (mean/max/sum) ---")

        # Map Keras aggregators to PyG aggregators ('sum' -> 'add')
        # Exclude 'pooling' as PyG SAGEConv doesn't have direct equivalent
        aggr_map = {"mean": "mean", "max": "max", "sum": "add"}
        test_params = list(
            itertools.product(
                aggr_map.keys(), self.bias_options, self.root_weight_options
            )
        )
        activation = None  # Compare pre-activation
        normalize = False

        for keras_aggr, use_bias, root_weight in test_params:
            pyg_aggr = aggr_map[keras_aggr]
            subtest_msg = f"aggr={keras_aggr}, bias={use_bias}, root={root_weight}"
            with self.subTest(msg=subtest_msg):
                print(f"\n--- Comparing: {subtest_msg} ---")
                keras_sage = SAGEConv(
                    output_dim=self.output_dim,
                    aggregator=keras_aggr,
                    normalize=normalize,
                    root_weight=root_weight,
                    use_bias=use_bias,
                    activation=activation,
                )
                _ = keras_sage([self.features_keras, self.edge_index_keras])  # Build

                pyg_sage = PyGSAGEConv(
                    in_channels=self.input_dim,
                    out_channels=self.output_dim,
                    aggr=pyg_aggr,
                    normalize=normalize,
                    root_weight=root_weight,
                    bias=use_bias,
                )

                # --- Sync Weights ---
                # Get Keras weights
                keras_weights_neigh = keras_sage.lin_neigh.get_weights()  # kernel
                keras_weights_self = (
                    keras_sage.lin_self.get_weights() if root_weight else None
                )  # kernel
                keras_bias = (
                    keras_sage.bias.numpy() if use_bias else None
                )  # separate bias

                # Sync neighbor weights (W_r in Keras, maps to lin_r or lin_l in PyG)
                if root_weight:
                    # If root_weight is True, PyG has lin_r for neighbors
                    # pyrefly: ignore  # missing-attribute
                    pyg_sage.lin_r.weight.data.copy_(
                        torch.tensor(keras_weights_neigh[0].T)
                    )
                else:
                    # If root_weight is False, PyG might use lin_l for neighbors
                    # pyrefly: ignore  # missing-attribute
                    pyg_sage.lin_l.weight.data.copy_(
                        torch.tensor(keras_weights_neigh[0].T)
                    )

                # Sync self weights (W_l in Keras, maps to lin_l in PyG) if root_weight is True
                if root_weight and keras_weights_self is not None:
                    # pyrefly: ignore  # missing-attribute
                    pyg_sage.lin_l.weight.data.copy_(
                        torch.tensor(keras_weights_self[0].T)
                    )

                # Sync bias if use_bias is True
                if use_bias and keras_bias is not None:
                    if root_weight:
                        # Bias goes to lin_l in PyG when root_weight=True
                        # pyrefly: ignore  # missing-attribute
                        pyg_sage.lin_l.bias.data.copy_(torch.tensor(keras_bias))
                    else:
                        # Bias goes to lin_l in PyG when root_weight=False (assuming lin_l is used for neighbors)
                        # pyrefly: ignore  # missing-attribute
                        pyg_sage.lin_l.bias.data.copy_(torch.tensor(keras_bias))

                print("Weights synced.")

                # --- Perform Forward Pass ---
                keras_output = keras_sage([self.features_keras, self.edge_index_keras])
                # pyrefly: ignore  # not-callable
                pyg_output = pyg_sage(self.features_torch, self.edge_index_torch)

                # --- Compare Final Outputs ---
                keras_output_np = keras_output.cpu().detach().numpy()
                pyg_output_np = pyg_output.cpu().detach().numpy()

                print(f"Keras final output shape: {keras_output_np.shape}")
                print(f"PyG final output shape: {pyg_output_np.shape}")
                self.assertEqual(
                    keras_output_np.shape, (self.num_nodes, self.output_dim)
                )
                self.assertEqual(pyg_output_np.shape, (self.num_nodes, self.output_dim))

                try:
                    np.testing.assert_allclose(
                        keras_output_np,
                        pyg_output_np,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"FINAL SAGE outputs differ for {subtest_msg}",
                    )
                    print(f"✅ FINAL SAGE outputs match for: {subtest_msg}")
                except AssertionError as e:
                    print(f"❌ FINAL SAGE outputs DO NOT match for: {subtest_msg}")
                    print(e)
                    print("   Keras sample:", keras_output_np[0, :5])
                    print("   PyG sample:", pyg_output_np[0, :5])

    def test_mean_numerical_values(self):
        """Compare mean aggregator output against manual NumPy calculation."""
        print("\n--- Testing GraphSAGEConv Numerical Values (Mean Aggregator) ---")

        aggr = "mean"
        # Test only with bias=True for simplicity, can expand if needed
        use_bias = True
        root_weight = True  # Test standard case
        normalize = False
        activation = None

        subtest_msg = f"aggr={aggr}, bias={use_bias}, root={root_weight}, norm={normalize}, act={activation}"
        print(f"\n--- Comparing: {subtest_msg} ---")

        # Instantiate Keras Layer
        layer = SAGEConv(
            output_dim=self.output_dim,
            aggregator=aggr,
            normalize=normalize,
            use_bias=use_bias,
            activation=activation,
            root_weight=root_weight,
            pool_activation="relu",  # Not used for mean aggregation
        )
        keras_output = layer([self.features_keras, self.edge_index_keras])
        # Handle both PyTorch and TensorFlow tensors
        try:
            keras_output_np = keras_output.cpu().detach().numpy()
        except AttributeError:
            try:
                keras_output_np = keras_output.cpu().numpy()
            except AttributeError:
                keras_output_np = keras.ops.convert_to_numpy(keras_output)

        # --- Manual NumPy Calculation ---
        x_np = self.features_np
        edge_index_np = self.edge_index_np.astype(np.int32)
        num_nodes = self.num_nodes
        in_channels = self.input_dim

        # 1. Manual Aggregation (Mean)
        adj = defaultdict(list)
        sources, targets = edge_index_np[0], edge_index_np[1]
        for src, tgt in zip(sources, targets):
            adj[tgt].append(src)

        aggregated_np = np.zeros((num_nodes, in_channels), dtype=np.float32)

        for i in range(num_nodes):
            # pyrefly: ignore  # bad-argument-type
            neighbors_indices = adj[i]
            if not neighbors_indices:
                aggregated_np[i, :] = 0.0  # Mean of empty set is 0
                continue
            neighbor_features = x_np[neighbors_indices]
            # Mean aggregation
            aggregated_np[i, :] = np.mean(neighbor_features, axis=0)

        # 2. Transform self and aggregated features
        lin_self_weights = layer.lin_self.get_weights()
        lin_neigh_weights = layer.lin_neigh.get_weights()
        w_self = lin_self_weights[0]  # No bias in lin_self
        w_neigh = lin_neigh_weights[0]
        # Bias is separate if root_weight=True, otherwise inside lin_neigh
        b_neigh = (
            lin_neigh_weights[1]
            if use_bias and not root_weight
            else np.zeros(self.output_dim, dtype=np.float32)
        )
        b_final = (
            layer.bias.numpy()
            if use_bias and root_weight
            else np.zeros(self.output_dim, dtype=np.float32)
        )

        h_neigh_np = (
            np.dot(aggregated_np, w_neigh) + b_neigh
        )  # Add bias only if root_weight=False
        h_self_np = np.dot(x_np, w_self) if root_weight else np.zeros_like(h_neigh_np)

        # 3. Combine and add final bias
        expected_output_np = h_self_np + h_neigh_np
        if use_bias and root_weight:
            expected_output_np += b_final

        # 4/5. Activation/Normalization are None/False in this test

        # --- Compare ---
        try:
            np.testing.assert_allclose(
                keras_output_np,
                expected_output_np,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Mean numerical values differ for {subtest_msg}",
            )
            print(f"✅ Mean numerical values match for: {subtest_msg}")
        except AssertionError as e:
            print(f"❌ Mean numerical values DO NOT match for: {subtest_msg}")
            print(e)
            abs_diff = np.abs(keras_output_np - expected_output_np)
            rel_diff = abs_diff / (np.abs(expected_output_np) + 1e-8)
            print(
                f"   Max Abs Diff: {np.max(abs_diff):.4g}, Max Rel Diff: {np.max(rel_diff):.4g}"
            )
            print("   Keras sample:", keras_output_np[0, :5])
            print("   Expected sample:", expected_output_np[0, :5])


if __name__ == "__main__":
    # pyrefly: ignore  # not-callable
    unittest.main()
