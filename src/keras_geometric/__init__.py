"""Keras Geometric: A Graph Neural Network Library for Keras."""

from ._version import __version__

# Layers
from .layers import GATv2Conv, GCNConv, GINConv, MessagePassing, SAGEConv

# Utilities
from .utils import GraphData, add_self_loops, batch_graphs, compute_gcn_normalization


# Datasets (when available)
# Use more specific import checks to avoid silent failures

def get_dataset_classes():
    try:
        from .datasets.citation import CiteSeer, PubMed
        from .datasets.cora import CoraDataset

        return {"CiteSeer": CiteSeer, "PubMed": PubMed, "Cora": CoraDataset}
    except ImportError as e:
        raise ImportError(f"Dataset dependencies not available: {e}") from e


# Define the __all__ list with all exported symbols
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