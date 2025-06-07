"""Attention-based pooling operations for graph-level representations."""

from typing import Any

import keras
from keras import initializers, layers, ops


class Set2Set(layers.Layer):
    """
    Set2Set pooling layer for creating graph-level representations.

    Implements the Set2Set pooling mechanism from "Order Matters: Sequence to sequence
    for sets" (https://arxiv.org/abs/1511.06391). This layer uses an LSTM-based
    attention mechanism to create permutation-invariant graph representations.

    Args:
        output_dim: Dimensionality of the output representation. The actual output
                   will be 2 * output_dim due to the LSTM's bidirectional nature.
        processing_steps: Number of processing steps for the attention mechanism.
                         More steps can lead to better representations but increased
                         computation. Defaults to 3.
        lstm_units: Number of LSTM units. If None, defaults to output_dim.
        dropout: Dropout rate for regularization. Defaults to 0.0.
        **kwargs: Additional arguments passed to the base Layer class.

    Example:
        ```python
        import keras
        import numpy as np
        from keras_geometric.layers.pooling import Set2Set

        # Create sample node features
        node_features = keras.ops.convert_to_tensor(
            np.random.randn(100, 64), dtype="float32"
        )

        # Create Set2Set pooling layer
        pool = Set2Set(output_dim=32, processing_steps=3)

        # Get graph-level representation
        graph_repr = pool(node_features)  # Shape: (1, 64) = (1, 2 * output_dim)
        ```

    Note:
        The output dimension is 2 * output_dim because Set2Set concatenates
        the final LSTM hidden state with the attention-weighted features.
    """

    def __init__(
        self,
        output_dim: int,
        processing_steps: int = 3,
        lstm_units: int | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if processing_steps <= 0:
            raise ValueError(
                f"processing_steps must be positive, got {processing_steps}"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.output_dim = output_dim
        self.processing_steps = processing_steps
        self.lstm_units = lstm_units if lstm_units is not None else output_dim
        self.dropout_rate = dropout

        # Initialize layers that will be built in build()
        self.lstm_cell: layers.LSTMCell | None = None
        self.attention_dense: layers.Dense | None = None
        self.dropout_layer: layers.Dropout | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer weights."""
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected input shape to be 2D (num_nodes, num_features), "
                f"got {len(input_shape)}D"
            )

        input_dim = input_shape[1]
        if input_dim is None:
            raise ValueError("Input feature dimension cannot be None")

        # LSTM cell for processing
        self.lstm_cell = layers.LSTMCell(
            self.lstm_units,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name="lstm_cell",
        )

        # Dense layer for attention computation
        self.attention_dense = layers.Dense(
            1,
            activation="tanh",
            kernel_initializer=initializers.get("glorot_uniform"),
            name="attention_dense",
        )

        # Dropout layer for regularization
        if self.dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(self.dropout_rate, name="dropout")

        super().build(input_shape)

    def call(  # pyrefly: ignore  # bad-override
        self, inputs: keras.KerasTensor, training: bool | None = None, **kwargs
    ) -> keras.KerasTensor:
        """
        Apply Set2Set pooling to node features.

        Args:
            inputs: Node features tensor of shape [num_nodes, num_features]
            training: Boolean indicating training or inference mode

        Returns:
            Graph-level representation tensor of shape [1, 2 * output_dim]
        """
        if self.lstm_cell is None or self.attention_dense is None:
            raise RuntimeError("Layer not built. Call build() first.")

        num_nodes = ops.shape(inputs)[0]
        # input_dim = ops.shape(inputs)[1]  # Unused variable

        # Initialize LSTM state
        # Hidden state and cell state both start as zeros
        h_state = ops.zeros((1, self.lstm_units), dtype=inputs.dtype)
        c_state = ops.zeros((1, self.lstm_units), dtype=inputs.dtype)

        # Process for the specified number of steps
        for _step in range(self.processing_steps):
            # Compute attention scores
            # Expand h_state to match all nodes: [num_nodes, lstm_units]
            # Squeeze h_state to remove batch dimension first
            h_squeezed = ops.squeeze(h_state, axis=0)  # [lstm_units]
            h_expanded = ops.broadcast_to(
                ops.expand_dims(h_squeezed, 0), (num_nodes, self.lstm_units)
            )  # [num_nodes, lstm_units]

            # Concatenate node features with LSTM hidden state
            attention_input = ops.concatenate([inputs, h_expanded], axis=-1)

            # Apply dropout if in training mode
            if self.dropout_layer is not None and training:
                attention_input = self.dropout_layer(attention_input, training=training)

            # Compute attention scores: [num_nodes, 1]
            attention_scores = self.attention_dense(attention_input)

            # Apply softmax to get attention weights: [num_nodes, 1]
            attention_weights = ops.softmax(attention_scores, axis=0)

            # Compute attention-weighted features: [1, input_dim]
            weighted_features = ops.sum(
                attention_weights * inputs, axis=0, keepdims=True
            )

            # Update LSTM state
            # LSTM cell expects and returns states as list [h, c]
            lstm_output = self.lstm_cell(
                weighted_features, states=[h_state, c_state], training=training
            )
            # Handle LSTM cell output: (output, [h_state, c_state])
            h_state, c_state = (
                lstm_output[1][0],
                lstm_output[1][1],
            )  # pyrefly: ignore  # bad-specialization

        # Final attention computation with the last hidden state
        h_squeezed = ops.squeeze(h_state, axis=0)  # [lstm_units]
        h_expanded = ops.broadcast_to(
            ops.expand_dims(h_squeezed, 0), (num_nodes, self.lstm_units)
        )  # [num_nodes, lstm_units]
        attention_input = ops.concatenate([inputs, h_expanded], axis=-1)

        if self.dropout_layer is not None and training:
            attention_input = self.dropout_layer(attention_input, training=training)

        attention_scores = self.attention_dense(attention_input)
        attention_weights = ops.softmax(attention_scores, axis=0)

        # Final weighted features
        final_weighted_features = ops.sum(
            attention_weights * inputs, axis=0, keepdims=True
        )

        # Concatenate LSTM hidden state with weighted features
        # Output dimension: [1, lstm_units + input_dim]
        output = ops.concatenate([h_state, final_weighted_features], axis=-1)

        return output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        """Compute the output shape given input shape."""
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected input shape to be 2D (num_nodes, num_features), "
                f"got {len(input_shape)}D"
            )

        input_dim = input_shape[1]
        if input_dim is None:
            raise ValueError("Input feature dimension cannot be None")

        # Output shape: [1, lstm_units + input_dim]
        return (1, self.lstm_units + input_dim)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "processing_steps": self.processing_steps,
                "lstm_units": self.lstm_units,
                "dropout": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Set2Set":
        """Create layer from configuration."""
        return cls(**config)


class AttentionPooling(layers.Layer):
    """
    Simple attention-based pooling layer.

    This layer computes attention weights over nodes and creates a weighted
    sum representation. It's simpler than Set2Set but still provides
    adaptive pooling based on node importance.

    Args:
        attention_dim: Dimensionality of the attention mechanism.
                      If None, uses input dimension.
        dropout: Dropout rate for regularization. Defaults to 0.0.
        **kwargs: Additional arguments passed to the base Layer class.

    Example:
        ```python
        import keras
        import numpy as np
        from keras_geometric.layers.pooling import AttentionPooling

        # Create sample node features
        node_features = keras.ops.convert_to_tensor(
            np.random.randn(100, 64), dtype="float32"
        )

        # Create attention pooling layer
        pool = AttentionPooling(attention_dim=32)

        # Get graph-level representation
        graph_repr = pool(node_features)  # Shape: (1, 64)
        ```
    """

    def __init__(
        self,
        attention_dim: int | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if attention_dim is not None and attention_dim <= 0:
            raise ValueError(f"attention_dim must be positive, got {attention_dim}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.attention_dim = attention_dim
        self.dropout_rate = dropout

        # Initialize layers that will be built in build()
        self.attention_dense: layers.Dense | None = None
        self.attention_score: layers.Dense | None = None
        self.dropout_layer: layers.Dropout | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer weights."""
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected input shape to be 2D (num_nodes, num_features), "
                f"got {len(input_shape)}D"
            )

        input_dim = input_shape[1]
        if input_dim is None:
            raise ValueError("Input feature dimension cannot be None")

        # Use input dimension if attention_dim not specified
        attention_dim = (
            self.attention_dim if self.attention_dim is not None else input_dim
        )

        # Dense layers for attention computation
        self.attention_dense = layers.Dense(
            attention_dim,
            activation="tanh",
            kernel_initializer=initializers.get("glorot_uniform"),
            name="attention_transform",
        )

        self.attention_score = layers.Dense(
            1,
            kernel_initializer=initializers.get("glorot_uniform"),
            name="attention_score",
        )

        # Dropout layer for regularization
        if self.dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(self.dropout_rate, name="dropout")

        super().build(input_shape)

    def call(  # pyrefly: ignore  # bad-override
        self, inputs: keras.KerasTensor, training: bool | None = None, **kwargs
    ) -> keras.KerasTensor:
        """
        Apply attention pooling to node features.

        Args:
            inputs: Node features tensor of shape [num_nodes, num_features]
            training: Boolean indicating training or inference mode

        Returns:
            Graph-level representation tensor of shape [1, num_features]
        """
        if self.attention_dense is None or self.attention_score is None:
            raise RuntimeError("Layer not built. Call build() first.")

        # Apply dropout if in training mode
        x = inputs
        if self.dropout_layer is not None and training:
            x = self.dropout_layer(x, training=training)

        # Compute attention features
        attention_features = self.attention_dense(x)

        # Compute attention scores: [num_nodes, 1]
        attention_scores = self.attention_score(attention_features)

        # Apply softmax to get attention weights: [num_nodes, 1]
        attention_weights = ops.softmax(attention_scores, axis=0)

        # Compute weighted sum: [1, num_features]
        pooled = ops.sum(attention_weights * inputs, axis=0, keepdims=True)

        return pooled

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        """Compute the output shape given input shape."""
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected input shape to be 2D (num_nodes, num_features), "
                f"got {len(input_shape)}D"
            )

        # Output shape is (1, num_features) - same as input features
        return (1, input_shape[1])

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "attention_dim": self.attention_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AttentionPooling":
        """Create layer from configuration."""
        return cls(**config)
