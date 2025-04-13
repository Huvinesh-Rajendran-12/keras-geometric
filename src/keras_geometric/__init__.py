from ._version import __version__
from .layers.gcn_conv import GCNConv
from .layers.gin_conv import GINConv
from .layers.message_passing import MessagePassing

__all__ = [
    "__version__",
    "GCNConv",
    "GINConv",
    "MessagePassing",
    # Add other exposed names here
]