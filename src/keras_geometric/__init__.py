"""Keras Geometric: A Graph Neural Network Library for Keras."""

from ._version import __version__

# Layers
from .layers.gatv2_conv import GATv2Conv
from .layers.gcn_conv import GCNConv
from .layers.gin_conv import GINConv
from .layers.message_passing import MessagePassing
from .layers.sage_conv import SAGEConv
from .utils.data_utils import GraphData, batch_graphs

# Utilities
from .utils.main import add_self_loops, compute_gcn_normalization

# Datasets (when available)
try:
    from .datasets.citation import CiteSeer, PubMed
    from .datasets.cora import CoraDataset
    __all__ = [
        "__version__",
        # Layers
        "GCNConv",
        "GINConv",
        "GATv2Conv",
        "SAGEConv",
        "MessagePassing",
        # Utilities
        "add_self_loops",
        "compute_gcn_normalization",
        "GraphData",
        "batch_graphs",
        # Datasets
        "CoraDataset",
        "CiteSeer",
        "PubMed",
    ]
except ImportError:
    __all__ = [
        "__version__",
        # Layers
        "GCNConv",
        "GINConv",
        "GATv2Conv",
        "SAGEConv",
        "MessagePassing",
        # Utilities
        "add_self_loops",
        "compute_gcn_normalization",
        "GraphData",
        "batch_graphs",
    ]
