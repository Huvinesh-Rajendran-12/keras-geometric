import unittest
import keras
import numpy as np
import sys
import os

# Check if the backend is set to 'torch'
assert keras.backend.backend() == 'torch', "Keras backend must be set to 'torch' for this test."

# Add the src directory to the path so we can import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import GCNConv using importlib to handle module name with hyphen
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gcn_conv", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'keras-geometric', 'gcn_conv.py')
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GCNConv = module.GCNConv

class TestGCNConv(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create small test data
        self.num_nodes = 4
        self.input_dim = 5
        self.output_dim = 3
        
        # Sample features and adjacency matrix
        self.features = np.random.random((self.num_nodes, self.input_dim)).astype(np.float32)
        # Create a simple adjacency matrix
        self.adjacency = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)

    def test_init(self):
        """Test initialization with different parameters"""
        # Test default initialization
        gcn = GCNConv(output_dim=self.output_dim)
        self.assertEqual(gcn.output_dim, self.output_dim)
        self.assertEqual(gcn.aggr, 'mean')
        self.assertTrue(gcn.use_bias)
        
        # Test with custom parameters
        gcn = GCNConv(output_dim=self.output_dim, aggr='sum', use_bias=False)
        self.assertEqual(gcn.output_dim, self.output_dim)
        self.assertEqual(gcn.aggr, 'sum')
        self.assertFalse(gcn.use_bias)
        
        # Test with invalid aggregation method
        with self.assertRaises(AssertionError):
            GCNConv(output_dim=self.output_dim, aggr='invalid')

    def test_call(self):
        """Test forward pass with different aggregation methods"""
        # Test with mean aggregation
        gcn_mean = GCNConv(output_dim=self.output_dim, aggr='mean')
        print(self.features.shape, self.adjacency.shape)
        output_mean = gcn_mean([self.features, self.adjacency])
        
        # Check output shape
        self.assertEqual(output_mean.shape, (self.num_nodes, self.output_dim))
        
        # Test with max aggregation
        gcn_max = GCNConv(output_dim=self.output_dim, aggr='max')
        output_max = gcn_max([self.features, self.adjacency])
        self.assertEqual(output_max.shape, (self.num_nodes, self.output_dim))
        
        # Test with add aggregation
        gcn_add = GCNConv(output_dim=self.output_dim, aggr='sum')
        output_add = gcn_add([self.features, self.adjacency])
        self.assertEqual(output_add.shape, (self.num_nodes, self.output_dim))
        
        # Test without bias
        gcn_no_bias = GCNConv(output_dim=self.output_dim, use_bias=False)
        output_no_bias = gcn_no_bias([self.features, self.adjacency])
        self.assertEqual(output_no_bias.shape, (self.num_nodes, self.output_dim))

    def test_get_config(self):
        """Test get_config method"""
        gcn = GCNConv(output_dim=self.output_dim, aggr='sum', use_bias=False)
        config = gcn.get_config()
        
        # Check that all the necessary keys are in the config
        self.assertIn('output_dim', config)
        self.assertIn('aggr', config)
        self.assertIn('use_bias', config)
        self.assertIn('kernel_initializer', config)
        self.assertIn('bias_initializer', config)
        
        # Check values
        self.assertEqual(config['output_dim'], self.output_dim)
        self.assertEqual(config['aggr'], 'sum')
        self.assertEqual(config['use_bias'], False)

if __name__ == '__main__':
    unittest.main()
