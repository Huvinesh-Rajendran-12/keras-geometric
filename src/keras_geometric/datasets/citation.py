import io
import os
import pickle
import urllib.request
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.data_utils import GraphData
from .base import Dataset


class CitationDataset(Dataset):
    """
    Base class for citation network datasets like Cora, CiteSeer, and PubMed.

    Citation networks are widely used benchmark datasets for node classification,
    where each node represents a scientific publication and edges represent citations.
    Node features are usually derived from the publication text.

    Args:
        root: Root directory where the dataset should be saved
        name: Name of the dataset ('cora', 'citeseer', 'pubmed')
        transform: A function/transform that takes in a GraphData object and returns
                  a transformed version
        pre_transform: A function/transform that takes in a GraphData object and returns
                       a transformed version, applied before the dataset is saved to disk

    Example:
        ```python
        # Load the Cora dataset
        dataset = CitationDataset(root="data", name="cora")

        # Get node features and labels
        graph = dataset[0]  # Only one graph in citation datasets
        x = graph.x  # Node features
        y = graph.y  # Node labels
        edge_index = graph.edge_index  # Citation edges
        ```
    """

    available_datasets = {
        'cora': {
            'url': 'https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.{}.pkl',
            'files': ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index'],
        },
        'citeseer': {
            'url': 'https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.{}.pkl',
            'files': ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index'],
        },
        'pubmed': {
            'url': 'https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.{}.pkl',
            'files': ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index'],
        }
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        # Validate dataset name
        if name.lower() not in self.available_datasets:
            raise ValueError(f"Dataset {name} not available. Choose from {list(self.available_datasets.keys())}")

        self.dataset_info = self.available_datasets[name.lower()]

        super().__init__(root, name.lower(), transform, pre_transform)

    def _download(self):
        """Download the dataset files."""
        os.makedirs(self._raw_dir(), exist_ok=True)

        # Download each file
        for file in self.dataset_info['files']:
            # Handle test.index specially, as it's not a pickle file
            if file == 'test.index':
                url = self.dataset_info['url'].format(file)
                target_path = os.path.join(self._raw_dir(), f"ind.{self.name}.{file}")
            else:
                url = self.dataset_info['url'].format(file)
                target_path = os.path.join(self._raw_dir(), f"ind.{self.name}.{file}.pkl")

            # Download if file doesn't exist
            if not os.path.exists(target_path):
                print(f"Downloading {url} to {target_path}")
                urllib.request.urlretrieve(url, target_path)

    def _load(self) -> Tuple[List[GraphData], int]:
        """
        Load the citation dataset.

        Returns:
            data_list: List containing a single GraphData object (the citation network)
            num_classes: Number of classes in the dataset
        """
        # Load raw data files
        raw_dir = self._raw_dir()

        # Load features and labels
        with open(os.path.join(raw_dir, f"ind.{self.name}.x.pkl"), 'rb') as f:
            x = pickle.load(f, encoding='latin1')
        with open(os.path.join(raw_dir, f"ind.{self.name}.y.pkl"), 'rb') as f:
            y = pickle.load(f, encoding='latin1')
        with open(os.path.join(raw_dir, f"ind.{self.name}.tx.pkl"), 'rb') as f:
            tx = pickle.load(f, encoding='latin1')
        with open(os.path.join(raw_dir, f"ind.{self.name}.ty.pkl"), 'rb') as f:
            ty = pickle.load(f, encoding='latin1')
        with open(os.path.join(raw_dir, f"ind.{self.name}.allx.pkl"), 'rb') as f:
            allx = pickle.load(f, encoding='latin1')
        with open(os.path.join(raw_dir, f"ind.{self.name}.ally.pkl"), 'rb') as f:
            ally = pickle.load(f, encoding='latin1')

        # Load graph structure
        with open(os.path.join(raw_dir, f"ind.{self.name}.graph.pkl"), 'rb') as f:
            graph_dict = pickle.load(f, encoding='latin1')

        # Load test indices
        with open(os.path.join(raw_dir, f"ind.{self.name}.test.index"), 'r') as f:
            test_idx = [int(i) for i in f.read().split()]

        # Combine features and labels
        x = np.vstack((allx.toarray(), tx.toarray())).astype(np.float32)
        y = np.vstack((ally, ty)).astype(np.int64)

        # Sort test indices
        test_idx_sorted = np.sort(test_idx)

        # Ensure y is in the correct order
        y_sorted = np.zeros_like(y)
        y_sorted[test_idx_sorted] = y[test_idx]
        y_sorted[~np.in1d(np.arange(y.shape[0]), test_idx_sorted)] = y[~np.in1d(np.arange(y.shape[0]), test_idx)]
        y = y_sorted

        # Convert graph dict to edge index
        edge_index = self._convert_graph_dict_to_edge_index(graph_dict)

        # Create GraphData object
        graph_data = GraphData(
            x=x,
            edge_index=edge_index,
            y=np.argmax(y, axis=1)  # Convert one-hot to class indices
        )

        # Number of classes
        num_classes = y.shape[1]

        return [graph_data], num_classes

    def _convert_graph_dict_to_edge_index(self, graph_dict: Dict[int, List[int]]) -> np.ndarray:
        """
        Convert a graph dictionary to an edge index matrix.

        Args:
            graph_dict: Dictionary where keys are source nodes and values are lists of target nodes

        Returns:
            edge_index: Edge index matrix with shape [2, num_edges]
        """
        # Collect edges
        edges = []
        for src, tgts in graph_dict.items():
            for tgt in tgts:
                # Add both directions for undirected graph
                edges.append((src, tgt))
                edges.append((tgt, src))

        # Remove duplicates and convert to numpy array
        edges = list(set(edges))
        edge_index = np.array(edges, dtype=np.int64).T

        return edge_index


class Cora(CitationDataset):
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
        dataset = Cora(root="data")

        # Get the single graph
        graph = dataset[0]

        # Get node features and labels
        x = graph.x  # Shape [2708, 1433]
        y = graph.y  # Shape [2708]
        edge_index = graph.edge_index  # Shape [2, 10858]
        ```
    """
    def __init__(
        self,
        root: str = "data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, "cora", transform, pre_transform)


class CiteSeer(CitationDataset):
    """
    The CiteSeer citation network dataset.

    Nodes represent scientific publications and edges represent citations.
    Node features are bag-of-words vectors of the publication text.
    The task is to classify each publication into one of 6 classes.

    Stats:
        - 3327 nodes
        - 4732 edges
        - 6 classes
        - 3703 features per node

    Example:
        ```python
        # Load the CiteSeer dataset
        dataset = CiteSeer(root="data")

        # Get the single graph
        graph = dataset[0]

        # Get node features and labels
        x = graph.x  # Shape [3327, 3703]
        y = graph.y  # Shape [3327]
        edge_index = graph.edge_index  # Shape [2, 9464]
        ```
    """
    def __init__(
        self,
        root: str = "data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, "citeseer", transform, pre_transform)


class PubMed(CitationDataset):
    """
    The PubMed citation network dataset.

    Nodes represent scientific publications and edges represent citations.
    Node features are bag-of-words vectors of the publication text.
    The task is to classify each publication into one of 3 classes.

    Stats:
        - 19717 nodes
        - 44338 edges
        - 3 classes
        - 500 features per node

    Example:
        ```python
        # Load the PubMed dataset
        dataset = PubMed(root="data")

        # Get the single graph
        graph = dataset[0]

        # Get node features and labels
        x = graph.x  # Shape [19717, 500]
        y = graph.y  # Shape [19717]
        edge_index = graph.edge_index  # Shape [2, 88676]
        ```
    """
    def __init__(
        self,
        root: str = "data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, "pubmed", transform, pre_transform)
