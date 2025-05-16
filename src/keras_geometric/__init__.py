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
# Use more specific import checks to avoid silent failures
dataset_imports_successful = True
try:
    from .datasets.citation import CiteSeer, PubMed
except ImportError:
    dataset_imports_successful = False

try:
    from .datasets.cora import CoraDataset
except ImportError:
    dataset_imports_successful = False

# Define the base __all__ list that's always available
base_all = [
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

# Add dataset imports if they succeeded
if dataset_imports_successful:
    __all__ = base_all + [
        # Datasets
        "CoraDataset",
        "CiteSeer",
        "PubMed",
    ]
else:
    __all__ = base_all
