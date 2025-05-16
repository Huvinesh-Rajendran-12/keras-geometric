import os

# import random  # Import the random module
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..utils.data_utils import GraphData


class Dataset:
    """
    Base class for graph datasets in Keras Geometric.

    This abstract class defines the interface for graph datasets. Subclasses
    should implement the _load method to load the actual data.

    Args:
        root: Root directory where the dataset should be saved
        name: Name of the dataset
        transform: A function/transform that takes in a GraphData object and returns
                  a transformed version
        pre_transform: A function/transform that takes in a GraphData object and returns
                       a transformed version, applied before the dataset is saved to disk

    Example:
        ```python
        class MyDataset(Dataset):
            def _load(self):
                # Load graph data here
                # ...

                # Return list of GraphData objects
                return graphs, num_classes

        dataset = MyDataset(root="data")
        train_data, val_data, test_data = dataset.split()
        ```
    """
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform

        self._data_list = None
        self._num_classes = None

        self._process()

    def _process(self):
        """Process the dataset and load it into memory."""
        # Create root directory if it doesn't exist
        os.makedirs(self.root, exist_ok=True)

        # Process or load processed data
        if self._is_processed():
            self._data_list, self._num_classes = self._load_processed()
        else:
            # Download raw data if needed
            if not self._is_raw_present():
                self._download()

            # Process raw data
            self._data_list, self._num_classes = self._process_raw()

            # Save processed data
            self._save_processed()

    def _is_processed(self) -> bool:
        """Check if the dataset has already been processed."""
        return os.path.exists(self._processed_file_path())

    def _is_raw_present(self) -> bool:
        """Check if raw data is present."""
        return os.path.exists(self._raw_dir())

    def _processed_file_path(self) -> str:
        """Path to the processed data file."""
        return os.path.join(self._processed_dir(), f"{self.name}.npz")

    def _processed_dir(self) -> str:
        """Directory for processed data."""
        return os.path.join(self.root, "processed")

    def _raw_dir(self) -> str:
        """Directory for raw data."""
        return os.path.join(self.root, "raw")

    def _download(self):
        """Download the dataset. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _download")

    def _process_raw(self) -> Tuple[List[GraphData], int]:
        """
        Process raw data into GraphData objects.

        Returns:
            data_list: List of GraphData objects
            num_classes: Number of classes in the dataset
        """
        # Create processed directory if it doesn't exist
        os.makedirs(self._processed_dir(), exist_ok=True)

        # Load and process raw data
        data_list, num_classes = self._load()

        # Apply pre_transform if specified
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list, num_classes

    def _save_processed(self):
        """Save processed data to disk."""
        # Prepare data for saving
        save_data = {}

        for i, graph_data in enumerate(self._data_list):
            # Convert GraphData to numpy arrays
            save_data[f"x_{i}"] = np.array(graph_data.x)
            save_data[f"edge_index_{i}"] = np.array(graph_data.edge_index)

            if graph_data.edge_attr is not None:
                save_data[f"edge_attr_{i}"] = np.array(graph_data.edge_attr)

            if graph_data.y is not None:
                save_data[f"y_{i}"] = np.array(graph_data.y)

        # Save metadata
        save_data["num_graphs"] = len(self._data_list)
        save_data["num_classes"] = self._num_classes

        # Save to file
        np.savez(self._processed_file_path(), **save_data)

    def _load_processed(self) -> Tuple[List[GraphData], int]:
        """Load processed data from disk."""
        # Load data from file
        data = np.load(self._processed_file_path(), allow_pickle=True)

        # Extract metadata
        num_graphs = int(data["num_graphs"])
        num_classes = int(data["num_classes"])

        # Reconstruct GraphData objects
        data_list = []
        for i in range(num_graphs):
            x = data[f"x_{i}"]
            edge_index = data[f"edge_index_{i}"]

            # Optional attributes
            edge_attr = data[f"edge_attr_{i}"] if f"edge_attr_{i}" in data else None
            y = data[f"y_{i}"] if f"y_{i}" in data else None

            # Create GraphData object
            graph_data = GraphData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )

            data_list.append(graph_data)

        return data_list, num_classes

    def _load(self) -> Tuple[List[GraphData], int]:
        """
        Load the dataset.

        This method should be implemented by subclasses to load the actual data.

        Returns:
            data_list: List of GraphData objects
            num_classes: Number of classes in the dataset
        """
        raise NotImplementedError("Subclasses must implement _load")

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[List[GraphData], List[GraphData], List[GraphData]]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle the data before splitting
            seed: Random seed for reproducibility

        Returns:
            train_data: List of GraphData objects for training
            val_data: List of GraphData objects for validation
            test_data: List of GraphData objects for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "The sum of train_ratio, val_ratio, and test_ratio must be 1"

        # Create a copy of the data list
        data_list: List[GraphData] = self._data_list if self._data_list is not None else []
        data_list = data_list.copy()

        # Shuffle data if needed
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(data_list)

        # Calculate split indices
        n = len(data_list)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split data
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]

        # Apply transform if specified
        if self.transform is not None:
            train_data = [self.transform(data) for data in train_data]
            val_data = [self.transform(data) for data in val_data]
            test_data = [self.transform(data) for data in test_data]

        return train_data, val_data, test_data

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        if self._data_list is None:
            return 0
        return len(self._data_list)

    def __getitem__(self, idx: int) -> GraphData:
        """Return the graph at index idx."""
        if self._data_list is None:
            raise IndexError("Dataset is empty")

        if idx < 0 or idx >= len(self._data_list):
            raise IndexError(f"Index {idx} out of range")

        data = self._data_list[idx]

        # Apply transform if specified
        if self.transform is not None:
            data = self.transform(data)

        return data
