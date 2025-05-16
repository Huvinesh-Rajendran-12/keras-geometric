import os
import sys
import time
import unittest

import keras
import numpy as np

# Add the source directory to the path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import modules from the package
from keras_geometric.layers import GCNConv, GINConv

# from keras_geometric.utils.data_utils import GraphData

# Try to import PyTorch Geometric for comparison
try:
    import torch

    # import torch_geometric
    from torch_geometric.nn import GCNConv as PyGGCNConv
    from torch_geometric.nn import GINConv as PyGGINConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and/or PyTorch Geometric not available. Skipping comparison benchmarks.")


class BenchmarkBase(unittest.TestCase):
    """Base class for benchmark tests."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

        # Define common test data sizes
        self.small_size = (100, 16)     # 100 nodes, 16 features
        self.medium_size = (1000, 64)   # 1k nodes, 64 features
        self.large_size = (10000, 128)  # 10k nodes, 128 features

        # Define number of edges for each size
        self.small_edges = 300
        self.medium_edges = 5000
        self.large_edges = 50000

        # Number of runs for averaging
        self.num_runs = 5

    def _generate_graph_data(self, num_nodes, num_features, num_edges):
        """Generate random graph data."""
        # Node features
        x = np.random.randn(num_nodes, num_features).astype(np.float32)

        # Random edges
        edge_index = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

        return x, edge_index

    def _measure_time(self, fn, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    def _run_benchmark(self, keras_fn, torch_fn=None, data_size="small",
                      output_dim=32, num_runs=None):
        """Run benchmark comparing Keras Geometric vs PyTorch Geometric."""
        if num_runs is None:
            num_runs = self.num_runs

        # Select data size
        if data_size == "small":
            num_nodes, num_features = self.small_size
            num_edges = self.small_edges
        elif data_size == "medium":
            num_nodes, num_features = self.medium_size
            num_edges = self.medium_edges
        elif data_size == "large":
            num_nodes, num_features = self.large_size
            num_edges = self.large_edges
        else:
            raise ValueError(f"Unknown data size: {data_size}")

        # Generate data
        x_np, edge_index_np = self._generate_graph_data(num_nodes, num_features, num_edges)

        # Keras setup
        x_keras = keras.ops.convert_to_tensor(x_np)
        edge_index_keras = keras.ops.convert_to_tensor(edge_index_np, dtype='int32')

        # Run Keras benchmark
        keras_times = []
        for _ in range(num_runs):
            time_taken, _ = self._measure_time(keras_fn, x_keras, edge_index_keras, output_dim)
            keras_times.append(time_taken)

        keras_avg_time = sum(keras_times) / len(keras_times)

        # Run PyTorch benchmark if available
        torch_avg_time = None
        if TORCH_AVAILABLE and torch_fn is not None:
            # PyTorch setup
            x_torch = torch.tensor(x_np)
            edge_index_torch = torch.tensor(edge_index_np, dtype=torch.int64)

            torch_times = []
            for _ in range(num_runs):
                time_taken, _ = self._measure_time(torch_fn, x_torch, edge_index_torch, output_dim)
                torch_times.append(time_taken)

            torch_avg_time = sum(torch_times) / len(torch_times)

        return {
            'keras_time': keras_avg_time,
            'torch_time': torch_avg_time,
            'data_size': data_size,
            'num_nodes': num_nodes,
            'num_features': num_features,
            'num_edges': num_edges,
            'output_dim': output_dim
        }


class TestGCNPerformance(BenchmarkBase):
    """Benchmark tests for GCNConv layer."""

    def _run_keras_gcn(self, x, edge_index, output_dim):
        """Run GCNConv forward pass."""
        gcn = GCNConv(output_dim=output_dim)
        return gcn([x, edge_index])

    def _run_torch_gcn(self, x, edge_index, output_dim):
        """Run PyTorch Geometric GCNConv forward pass."""
        if not TORCH_AVAILABLE:
            return None
        gcn = PyGGCNConv(in_channels=x.size(1), out_channels=output_dim)
        return gcn(x, edge_index)

    def test_gcn_small(self):
        """Benchmark GCNConv on small graphs."""
        results = self._run_benchmark(
            self._run_keras_gcn,
            self._run_torch_gcn if TORCH_AVAILABLE else None,
            "small"
        )

        print(f"\nGCNConv Small Benchmark Results:")
        print(f"  Data: {results['num_nodes']} nodes, {results['num_features']} features, {results['num_edges']} edges")
        print(f"  Keras Geometric: {results['keras_time']:.6f} seconds")
        if results['torch_time']:
            print(f"  PyTorch Geometric: {results['torch_time']:.6f} seconds")
            print(f"  Ratio (PyG/KerasG): {results['torch_time']/results['keras_time']:.2f}x")

    def test_gcn_medium(self):
        """Benchmark GCNConv on medium graphs."""
        results = self._run_benchmark(
            self._run_keras_gcn,
            self._run_torch_gcn if TORCH_AVAILABLE else None,
            "medium"
        )

        print(f"\nGCNConv Medium Benchmark Results:")
        print(f"  Data: {results['num_nodes']} nodes, {results['num_features']} features, {results['num_edges']} edges")
        print(f"  Keras Geometric: {results['keras_time']:.6f} seconds")
        if results['torch_time']:
            print(f"  PyTorch Geometric: {results['torch_time']:.6f} seconds")
            print(f"  Ratio (PyG/KerasG): {results['torch_time']/results['keras_time']:.2f}x")


class TestGINPerformance(BenchmarkBase):
    """Benchmark tests for GINConv layer."""

    def _run_keras_gin(self, x, edge_index, output_dim):
        """Run GINConv forward pass."""
        # Create a GIN layer with a simple MLP
        gin = GINConv(output_dim=output_dim, mlp_hidden=[output_dim*2])
        return gin([x, edge_index])

    def _run_torch_gin(self, x, edge_index, output_dim):
        """Run PyTorch Geometric GINConv forward pass."""
        if not TORCH_AVAILABLE:
            return None

        # Create a simple MLP for the GINConv
        nn_model = torch.nn.Sequential(
            torch.nn.Linear(x.size(1), output_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim*2, output_dim)
        )
        gin = PyGGINConv(nn=nn_model)
        return gin(x, edge_index)

    def test_gin_small(self):
        """Benchmark GINConv on small graphs."""
        results = self._run_benchmark(
            self._run_keras_gin,
            self._run_torch_gin if TORCH_AVAILABLE else None,
            "small"
        )

        print(f"\nGINConv Small Benchmark Results:")
        print(f"  Data: {results['num_nodes']} nodes, {results['num_features']} features, {results['num_edges']} edges")
        print(f"  Keras Geometric: {results['keras_time']:.6f} seconds")
        if results['torch_time']:
            print(f"  PyTorch Geometric: {results['torch_time']:.6f} seconds")
            print(f"  Ratio (PyG/KerasG): {results['torch_time']/results['keras_time']:.2f}x")

    def test_gin_medium(self):
        """Benchmark GINConv on medium graphs."""
        results = self._run_benchmark(
            self._run_keras_gin,
            self._run_torch_gin if TORCH_AVAILABLE else None,
            "medium"
        )

        print(f"\nGINConv Medium Benchmark Results:")
        print(f"  Data: {results['num_nodes']} nodes, {results['num_features']} features, {results['num_edges']} edges")
        print(f"  Keras Geometric: {results['keras_time']:.6f} seconds")
        if results['torch_time']:
            print(f"  PyTorch Geometric: {results['torch_time']:.6f} seconds")
            print(f"  Ratio (PyG/KerasG): {results['torch_time']/results['keras_time']:.2f}x")


if __name__ == '__main__':
    unittest.main()
