from typing import Any, Optional, Union

import keras
import numpy as np
from keras import ops


class GraphData:
    """
    A data structure for storing and managing graph data.

    This class provides a consistent interface for managing node features, edge indices,
    edge features, and other graph-related data for use with GNN layers.

    Args:
        x: Node features matrix with shape [num_nodes, num_node_features]
        edge_index: Edge index matrix with shape [2, num_edges]
        edge_attr: Optional edge feature matrix with shape [num_edges, num_edge_features]
        y: Optional node-level or graph-level target with arbitrary shape
        num_nodes: Optional explicit number of nodes (useful for isolated nodes)
        **kwargs: Additional data to store

    Example:
        ```python
        # Create a simple graph with 3 nodes and 2 edges
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # 3 nodes with 2 features each
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int32)   # Edges: 0->1, 1->2

        graph = GraphData(x=x, edge_index=edge_index)

        # Access data
        x = graph.x
        edge_index = graph.edge_index
        ```
    """

    def __init__(
        self,
        x: Union[np.ndarray, "keras.KerasTensor"],
        edge_index: Union[np.ndarray, "keras.KerasTensor"],
        edge_attr: Optional[Union[np.ndarray, "keras.KerasTensor"]] = None,
        y: Optional[Union[np.ndarray, "keras.KerasTensor"]] = None,
        num_nodes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Convert numpy arrays to tensors if needed
        self.x = self._ensure_tensor(x)
        self.edge_index = self._ensure_tensor(edge_index, dtype="int32")

        # Store optional attributes
        self.edge_attr = (
            self._ensure_tensor(edge_attr) if edge_attr is not None else None
        )
        self.y = self._ensure_tensor(y) if y is not None else None

        # Store number of nodes
        if num_nodes is None:
            self._num_nodes = ops.shape(self.x)[0]
        else:
            self._num_nodes = num_nodes

        # Store any additional data
        self._additional_data = {}
        for key, value in kwargs.items():
            self._additional_data[key] = self._ensure_tensor(value)

    def _ensure_tensor(self, data: Any, dtype: Optional[str] = None) -> Any:
        """Convert data to a tensor if it's not already one."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return keras.ops.convert_to_tensor(data, dtype=dtype)
        return data

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph."""
        if self.edge_index is None:
            return 0
        return ops.shape(self.edge_index)[1]

    @property
    def num_node_features(self) -> int:
        """Get the number of node features."""
        if self.x is None:
            return 0
        return ops.shape(self.x)[1]

    @property
    def num_edge_features(self) -> int:
        """Get the number of edge features."""
        if self.edge_attr is None:
            return 0
        return ops.shape(self.edge_attr)[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph data to a dictionary."""
        data_dict = {
            "x": self.x,
            "edge_index": self.edge_index,
        }

        if self.edge_attr is not None:
            data_dict["edge_attr"] = self.edge_attr

        if self.y is not None:
            data_dict["y"] = self.y

        data_dict.update(self._additional_data)

        return data_dict

    def to_inputs(self) -> list:
        """
        Convert the graph data to a list of inputs for use with Keras models.
        Useful for functional API models.
        """
        inputs = [self.x, self.edge_index]

        if self.edge_attr is not None:
            inputs.append(self.edge_attr)

        return inputs

    def __getattr__(self, name: str) -> Any:
        """Access additional data by attribute name."""
        if name in self._additional_data:
            return self._additional_data[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def batch_graphs(graphs: list[GraphData]) -> GraphData:
    """
    Batch multiple graphs into a single large graph with disjoint components.

    This function combines multiple GraphData objects into a single GraphData object
    by shifting node indices and concatenating features.

    Args:
        graphs: List of GraphData objects to batch

    Returns:
        A single GraphData object representing the batched graph

    Example:
        ```python
        # Create two small graphs
        graph1 = GraphData(x=x1, edge_index=edge_index1)
        graph2 = GraphData(x=x2, edge_index=edge_index2)

        # Batch them
        batched_graph = batch_graphs([graph1, graph2])
        ```
    """
    if not graphs:
        raise ValueError("Cannot batch empty list of graphs")

    # Get dimensions for pre-allocation
    total_nodes = sum(g.num_nodes for g in graphs)
    total_edges = sum(g.num_edges for g in graphs)

    # Pre-allocate arrays with safe dtype checks
    # batch_x = ops.zeros((total_nodes, node_feature_dim), dtype=x_dtype)

    x_list = [g.x for g in graphs]
    batch_x = ops.concatenate(x_list, axis=0)

    edge_index_dtype = (
        graphs[0].edge_index.dtype
        if graphs[0].edge_index is not None and hasattr(graphs[0].edge_index, "dtype")
        else "int32"
    )
    batch_edge_index = ops.zeros((2, total_edges), dtype=edge_index_dtype)

    # Track batch indices for each node
    batch_indices = ops.zeros((total_nodes,), dtype="int32")

    has_edge_attr = all(g.edge_attr is not None for g in graphs)
    if has_edge_attr:
        edge_feature_dim = graphs[0].num_edge_features
        edge_attr_dtype = (
            graphs[0].edge_attr.dtype
            if graphs[0].edge_attr is not None and hasattr(graphs[0].edge_attr, "dtype")
            else "float32"
        )
        batch_edge_attr = ops.zeros(
            (total_edges, edge_feature_dim), dtype=edge_attr_dtype
        )
    else:
        batch_edge_attr = None

    has_y = all(g.y is not None for g in graphs)
    if has_y:
        # Assume all targets have the same shape
        y_shape = ops.shape(graphs[0].y)
        y_dtype = (
            graphs[0].y.dtype
            if graphs[0].y is not None and hasattr(graphs[0].y, "dtype")
            else "float32"
        )
        if len(y_shape) == 1:  # Graph-level target
            batch_y = ops.zeros((len(graphs), y_shape[0]), dtype=y_dtype)
        else:
            # Assume node-level targets
            batch_y = ops.zeros((total_nodes, y_shape[1]), dtype=y_dtype)
    else:
        batch_y = None

    # Combine graphs
    node_offset = 0
    edge_offset = 0

    for i, graph in enumerate(graphs):
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges

        # Copy node features
        batch_x = ops.slice_update(batch_x, [node_offset, 0], graph.x)

        # Set batch indices for these nodes
        batch_indices = ops.slice_update(
            batch_indices,
            [node_offset],
            ops.multiply(ops.ones_like(ops.arange(num_nodes), dtype="int32"), i),
        )

        # Copy and adjust edge indices
        if num_edges > 0:
            shifted_edge_index = ops.add(graph.edge_index, node_offset)
            batch_edge_index = ops.slice_update(
                batch_edge_index, [0, edge_offset], shifted_edge_index
            )

            # Copy edge features if available
            if has_edge_attr:
                batch_edge_attr = ops.slice_update(
                    batch_edge_attr, [edge_offset, 0], graph.edge_attr
                )

        # Copy targets if available
        if has_y:
            y_shape = ops.shape(graph.y)
            if len(y_shape) == 1:  # Graph-level target
                batch_y = ops.slice_update(
                    batch_y, [i, 0], ops.expand_dims(graph.y, axis=0)
                )
            else:
                # Assume node-level targets
                batch_y = ops.slice_update(batch_y, [node_offset, 0], graph.y)

        # Update offsets
        node_offset += num_nodes
        edge_offset += num_edges

    # Create batched graph
    batched_graph = GraphData(
        x=batch_x,
        edge_index=batch_edge_index,
        edge_attr=batch_edge_attr,
        y=batch_y,
        num_nodes=total_nodes,
        batch=batch_indices,
    )

    return batched_graph
