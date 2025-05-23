import os
import tarfile
import urllib.request
from typing import Callable, Optional

import numpy as np

try:
    import scipy.sparse as sp
except ImportError as err:
    raise ImportError(
        "scipy is required for loading the Cora dataset. "
        "You can install it with `pip install scipy`."
    ) from err

from ..utils.data_utils import GraphData
from .base import Dataset


class CoraDataset(Dataset):
    """
    The Cora citation network dataset.

    Nodes represent scientific publications and edges represent citations.
    Node features are bag-of-words vectors of the publication text.
    The task is to classify each publication into one of 7 classes.

    Stats:
        - 2708 nodes
        - 5429 edges
        - 7 classes
        - 1433 features per node

    Example:
        ```python
        # Load the Cora dataset
        dataset = CoraDataset(root="data")

        # Get the single graph
        graph = dataset[0]

        # Get node features and labels
        x = graph.x  # Shape [2708, 1433]
        y = graph.y  # Shape [2708]
        edge_index = graph.edge_index  # Shape [2, 10858]
        ```
    """

    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

    def __init__(
        self,
        root: str = "data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, "cora", transform, pre_transform)

    def _download(self):
        """Download the dataset files."""
        os.makedirs(self._raw_dir(), exist_ok=True)

        # Download tgz file
        tgz_path = os.path.join(self._raw_dir(), "cora.tgz")
        if not os.path.exists(tgz_path):
            print(f"Downloading {self.url}")
            urllib.request.urlretrieve(self.url, tgz_path)

        # Extract files
        if not os.path.exists(os.path.join(self._raw_dir(), "cora")):
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(self._raw_dir())

    def _load(self) -> tuple[list[GraphData], int]:
        """
        Load the Cora dataset.

        Returns:
            data_list: List containing a single GraphData object (the citation network)
            num_classes: Number of classes in the dataset
        """
        raw_dir = os.path.join(self._raw_dir(), "cora")

        # Load content file (node features and labels)
        content_path = os.path.join(raw_dir, "cora.content")
        content = np.genfromtxt(content_path, dtype=np.dtype(str))

        # Extract features and labels
        features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32).toarray()
        labels = self._encode_labels(content[:, -1])

        # Create node ID mapping
        id_mapping = {id_: idx for idx, id_ in enumerate(content[:, 0])}

        # Load citations (edges)
        cites_path = os.path.join(raw_dir, "cora.cites")
        cites = np.genfromtxt(cites_path, dtype=np.dtype(str))

        # Convert citation IDs to indices
        edge_list: list[list[int]] = []
        for edge in cites:
            source_idx = id_mapping.get(edge[0])
            target_idx = id_mapping.get(edge[1])
            if source_idx is not None and target_idx is not None:
                edge_list.append([source_idx, target_idx])
                edge_list.append(
                    [target_idx, source_idx]
                )  # Add reverse edge for undirected graph

        edge_index = np.array(edge_list, dtype=np.int64).T

        # Create GraphData object
        graph_data = GraphData(x=features, edge_index=edge_index, y=labels)

        return [graph_data], len(np.unique(labels))

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert string labels to integer indices."""
        unique_labels = np.unique(labels)
        label_dict = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array([label_dict[label] for label in labels], dtype=np.int64)
