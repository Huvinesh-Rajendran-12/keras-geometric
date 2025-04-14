from ._version import __version__
from .layers.gcn_conv import GCNConv
from .layers.gin_conv import GINConv
from .layers.message_passing import MessagePassing
from .utils.main import add_self_loops, compute_gcn_normalization

__all__ = [
    "__version__",
    "GCNConv",
    "GINConv",
    "MessagePassing",
    "add_self_loops",
    "compute_gcn_normalization",
    # Add other exposed names here
]