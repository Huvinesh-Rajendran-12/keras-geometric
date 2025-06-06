"""
Aggregation strategies for message passing layers.

This module implements the Strategy pattern for message aggregation in Graph Neural Networks.
Each aggregator encapsulates a specific aggregation algorithm, making the code more modular,
testable, and extensible.
"""

from abc import ABC, abstractmethod

import keras
from keras import ops
from keras.src.backend.common.keras_tensor import KerasTensor


class Aggregator(ABC):
    """
    Abstract base class for message aggregation strategies.

    Aggregators define how messages from neighboring nodes are combined
    to update target node representations in Graph Neural Networks.
    """

    @abstractmethod
    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        """
        Aggregate messages for target nodes.

        Args:
            messages: Tensor of shape [E, F] containing messages from source nodes.
            target_idx: Tensor of shape [E] containing target node indices for each message.
            dim_size: Total number of nodes in the graph.

        Returns:
            Tensor of shape [N, F] containing aggregated messages for each node.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this aggregator."""
        pass


class MeanAggregator(Aggregator):
    """
    Mean aggregation: computes the average of incoming messages.

    For node i: aggr_i = (1/|N(i)|) * Σ_{j ∈ N(i)} m_{j→i}
    where N(i) is the set of neighbors of node i.
    """

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        # Count the number of messages per node
        ones = ops.ones((ops.shape(messages)[0], 1), dtype=messages.dtype)
        degree = ops.segment_sum(
            data=ones, segment_ids=target_idx, num_segments=dim_size
        )

        # Sum messages per node
        aggregated_sum = ops.segment_sum(
            data=messages, segment_ids=target_idx, num_segments=dim_size
        )

        # Compute mean with numerical stability
        epsilon = ops.convert_to_tensor(1e-8, dtype=degree.dtype)
        degree = ops.maximum(degree, epsilon)

        # Compute mean
        aggregated_mean = aggregated_sum / degree

        # For nodes with no incoming edges, the result should be zeros
        # This is already handled by segment_sum returning zeros
        return aggregated_mean

    @property
    def name(self) -> str:
        return "mean"


class MaxAggregator(Aggregator):
    """
    Max aggregation: computes the element-wise maximum of incoming messages.

    For node i: aggr_i = max_{j ∈ N(i)} m_{j→i}
    """

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        aggr = ops.segment_max(
            data=messages, segment_ids=target_idx, num_segments=dim_size
        )
        # Replace -inf values with zeros (for nodes with no incoming messages)
        return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)

    @property
    def name(self) -> str:
        return "max"


class SumAggregator(Aggregator):
    """
    Sum aggregation: computes the sum of incoming messages.

    For node i: aggr_i = Σ_{j ∈ N(i)} m_{j→i}
    """

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        return ops.segment_sum(
            data=messages, segment_ids=target_idx, num_segments=dim_size
        )

    @property
    def name(self) -> str:
        return "sum"


class MinAggregator(Aggregator):
    """
    Min aggregation: computes the element-wise minimum of incoming messages.

    For node i: aggr_i = min_{j ∈ N(i)} m_{j→i}
    """

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        # Compute min using negative max
        negative_data = ops.negative(messages)
        aggr = ops.segment_max(
            data=negative_data, segment_ids=target_idx, num_segments=dim_size
        )
        aggr = ops.negative(aggr)
        # Replace inf values with zeros (for nodes with no incoming messages)
        return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)

    @property
    def name(self) -> str:
        return "min"


class StdAggregator(Aggregator):
    """
    Standard deviation aggregation: computes the standard deviation of incoming messages.

    For node i: aggr_i = sqrt(Var[m_{j→i} : j ∈ N(i)])
    Uses Welford's algorithm for numerical stability.
    """

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            feature_dim = ops.shape(messages)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        # Compute standard deviation aggregation using Welford's algorithm for stability
        # First, compute the mean
        ones = ops.ones((ops.shape(messages)[0], 1), dtype=messages.dtype)
        count = ops.segment_sum(
            data=ones, segment_ids=target_idx, num_segments=dim_size
        )

        sum_messages = ops.segment_sum(
            data=messages, segment_ids=target_idx, num_segments=dim_size
        )

        # Safe mean computation
        epsilon = ops.convert_to_tensor(1e-8, dtype=count.dtype)
        safe_count = ops.maximum(count, epsilon)
        mean = sum_messages / safe_count

        # Expand mean to match messages for each edge
        mean_expanded = ops.take(mean, target_idx, axis=0)

        # Compute squared differences - ensure both operands are tensors
        messages_tensor = ops.convert_to_tensor(messages)
        squared_diff = ops.square(messages_tensor - mean_expanded)

        # Sum squared differences per node
        sum_squared_diff = ops.segment_sum(
            data=squared_diff, segment_ids=target_idx, num_segments=dim_size
        )

        # Compute variance (using N instead of N-1 for consistency)
        variance = sum_squared_diff / safe_count

        # Compute standard deviation with numerical stability
        std_dev = ops.sqrt(ops.maximum(variance, ops.zeros_like(variance)))

        # For nodes with single or no neighbors, std should be 0
        std_dev = ops.where(count <= 1, ops.zeros_like(std_dev), std_dev)

        return std_dev

    @property
    def name(self) -> str:
        return "std"


class PoolingAggregator(Aggregator):
    """
    Pooling aggregation: applies max pooling after a learnable transformation.

    For node i: aggr_i = max_{j ∈ N(i)} MLP(m_{j→i})
    where MLP is a multi-layer perceptron transformation.

    Note: This aggregator requires a transformation layer to be provided.
    """

    def __init__(self, pool_mlp: keras.Layer) -> None:
        """
        Initialize pooling aggregator with a transformation layer.

        Args:
            pool_mlp: Keras layer to transform messages before pooling.
        """
        self.pool_mlp = pool_mlp

    def aggregate(
        self, messages: KerasTensor, target_idx: KerasTensor, dim_size: int
    ) -> KerasTensor:
        if ops.shape(messages)[0] == 0:
            # Need to determine output dimension from the MLP
            dummy_input = ops.zeros((1, ops.shape(messages)[1]), dtype=messages.dtype)
            dummy_output = self.pool_mlp(dummy_input)
            feature_dim = ops.shape(dummy_output)[1]
            return ops.zeros((dim_size, feature_dim), dtype=messages.dtype)

        target_idx = ops.cast(target_idx, dtype="int32")

        # Transform messages through MLP
        transformed_messages = self.pool_mlp(messages)

        # Apply max pooling
        aggr = ops.segment_max(
            data=transformed_messages, segment_ids=target_idx, num_segments=dim_size
        )
        # Replace -inf values with zeros (for nodes with no incoming messages)
        return ops.where(ops.isinf(aggr), ops.zeros_like(aggr), aggr)

    @property
    def name(self) -> str:
        return "pooling"


class AggregatorFactory:
    """
    Factory class for creating aggregator instances.

    This factory provides a centralized way to create aggregators and ensures
    type safety by validating aggregator names at creation time.
    """

    _AGGREGATORS: dict[str, type[Aggregator]] = {
        "mean": MeanAggregator,
        "max": MaxAggregator,
        "sum": SumAggregator,
        "min": MinAggregator,
        "std": StdAggregator,
    }

    @classmethod
    def create(cls, aggregator_name: str, **kwargs) -> Aggregator:
        """
        Create an aggregator instance by name.

        Args:
            aggregator_name: Name of the aggregator to create.
            **kwargs: Additional arguments to pass to the aggregator constructor.

        Returns:
            Aggregator instance.

        Raises:
            ValueError: If aggregator_name is not supported.
        """
        if aggregator_name not in cls._AGGREGATORS:
            available = list(cls._AGGREGATORS.keys())
            raise ValueError(
                f"Invalid aggregator: {aggregator_name}. "
                f"Available aggregators: {available}"
            )

        aggregator_class = cls._AGGREGATORS[aggregator_name]
        return aggregator_class(**kwargs)

    @classmethod
    def create_pooling(cls, pool_mlp: keras.Layer) -> PoolingAggregator:
        """
        Create a pooling aggregator with the specified MLP.

        Args:
            pool_mlp: Keras layer to use for message transformation.

        Returns:
            PoolingAggregator instance.
        """
        return PoolingAggregator(pool_mlp)

    @classmethod
    def get_available_aggregators(cls) -> list[str]:
        """
        Get list of available aggregator names.

        Returns:
            List of supported aggregator names.
        """
        return list(cls._AGGREGATORS.keys())
