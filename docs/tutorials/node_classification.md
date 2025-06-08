# Node Classification Tutorial

Node classification is one of the most fundamental tasks in graph machine learning. In this tutorial, you'll learn how to classify nodes in a graph based on their features and neighborhood structure.

## What is Node Classification?

Node classification involves predicting labels for nodes in a graph based on:
- **Node features**: Attributes of individual nodes
- **Graph structure**: How nodes are connected
- **Neighborhood information**: Features of neighboring nodes

Common applications include:
- **Social networks**: Predicting user interests or demographics
- **Citation networks**: Classifying research papers by topic
- **Biological networks**: Predicting protein functions
- **Knowledge graphs**: Entity type classification

## Dataset: Citation Network

We'll use a citation network where:
- **Nodes**: Research papers
- **Edges**: Citation relationships
- **Node features**: Bag-of-words representation of paper abstracts
- **Task**: Predict research area (e.g., AI, Theory, Systems)

```python
import numpy as np
import keras
from keras_geometric.layers import GCNConv, GATv2Conv, SAGEConv
from keras_geometric.datasets import CoraDataset

# Set backend
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load Cora dataset
print("Loading Cora dataset...")
dataset = CoraDataset()
data = dataset[0]  # Single graph

print(f"Dataset: {dataset}")
print(f"Number of nodes: {data.x.shape[0]}")
print(f"Number of edges: {data.edge_index.shape[1]}")
print(f"Number of features: {data.x.shape[1]}")
print(f"Number of classes: {len(np.unique(data.y))}")
```

## Building Your First Node Classifier

### Simple GCN Model

Let's start with a basic Graph Convolutional Network:

```python
class SimpleGCN(keras.Model):
    \"\"\"Simple 2-layer GCN for node classification.\"\"\"

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNConv(hidden_dim, use_bias=True)
        self.gcn2 = GCNConv(num_classes, use_bias=True)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        # First layer
        x = self.gcn1([x, edge_index])
        x = keras.ops.relu(x)
        x = self.dropout(x, training=training)

        # Second layer
        x = self.gcn2([x, edge_index])
        return x

# Create model
model = SimpleGCN(
    input_dim=data.x.shape[1],
    hidden_dim=64,
    num_classes=len(np.unique(data.y))
)

# Forward pass
logits = model([data.x, data.edge_index], training=False)
predictions = keras.ops.softmax(logits)

print(f"Output shape: {predictions.shape}")
print(f"Predictions sum to 1: {np.allclose(keras.ops.sum(predictions, axis=1), 1.0)}")
```

### Training Loop

Now let's implement a complete training loop:

```python
def train_node_classifier(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    \"\"\"Train a node classification model.\"\"\"

    # Create optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)

    # Convert data to tensors
    x = keras.ops.convert_to_tensor(data.x, dtype="float32")
    edge_index = keras.ops.convert_to_tensor(data.edge_index, dtype="int32")
    y = keras.ops.convert_to_tensor(data.y, dtype="int32")

    # Create train/val/test masks (if not provided)
    num_nodes = x.shape[0]
    if not hasattr(data, 'train_mask'):
        # Simple split: 60% train, 20% val, 20% test
        indices = np.random.permutation(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    train_mask = keras.ops.convert_to_tensor(train_mask)
    val_mask = keras.ops.convert_to_tensor(val_mask)
    test_mask = keras.ops.convert_to_tensor(test_mask)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(epochs):
        # Training step
        with keras.utils.custom_object_scope():
            # Forward pass
            logits = model([x, edge_index], training=True)

            # Compute loss (only on training nodes)
            train_logits = keras.ops.boolean_mask(logits, train_mask)
            train_labels = keras.ops.boolean_mask(y, train_mask)

            loss = keras.ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    train_labels, train_logits, from_logits=True
                )
            )

        # Evaluation
        with keras.utils.custom_object_scope():
            logits = model([x, edge_index], training=False)
            pred = keras.ops.argmax(logits, axis=1)

            # Validation accuracy
            val_pred = keras.ops.boolean_mask(pred, val_mask)
            val_labels = keras.ops.boolean_mask(y, val_mask)
            val_acc = keras.ops.mean(keras.ops.equal(val_pred, val_labels))

            # Test accuracy
            test_pred = keras.ops.boolean_mask(pred, test_mask)
            test_labels = keras.ops.boolean_mask(y, test_mask)
            test_acc = keras.ops.mean(keras.ops.equal(test_pred, test_labels))

        # Track best performance
        val_acc_val = keras.ops.convert_to_numpy(val_acc)
        test_acc_val = keras.ops.convert_to_numpy(test_acc)

        if val_acc_val > best_val_acc:
            best_val_acc = val_acc_val
            best_test_acc = test_acc_val

        # Print progress
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {keras.ops.convert_to_numpy(loss):.4f} | "
                  f"Val Acc: {val_acc_val:.4f} | Test Acc: {test_acc_val:.4f}")

    print(f"\\nBest Test Accuracy: {best_test_acc:.4f}")
    return best_test_acc

# Train the model
print("Training GCN...")
gcn_accuracy = train_node_classifier(model, data)
```

## Advanced Models

### Graph Attention Network (GAT)

GATs use attention mechanisms to learn the importance of different neighbors:

```python
class GATNodeClassifier(keras.Model):
    \"\"\"Graph Attention Network for node classification.\"\"\"

    def __init__(self, input_dim, hidden_dim, num_classes, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATv2Conv(hidden_dim, heads=heads, dropout=dropout, use_bias=True)
        self.gat2 = GATv2Conv(num_classes, heads=1, dropout=dropout, use_bias=True)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        # First GAT layer (multi-head)
        x = self.gat1([x, edge_index])
        x = keras.ops.elu(x)
        x = self.dropout(x, training=training)

        # Second GAT layer (single head)
        x = self.gat2([x, edge_index])
        return x

# Create and train GAT model
print("\\nTraining GAT...")
gat_model = GATNodeClassifier(
    input_dim=data.x.shape[1],
    hidden_dim=8,  # Smaller hidden dim due to multi-head
    num_classes=len(np.unique(data.y)),
    heads=8
)

gat_accuracy = train_node_classifier(gat_model, data)
```

### GraphSAGE Model

GraphSAGE is particularly good for large graphs and inductive learning:

```python
class SAGENodeClassifier(keras.Model):
    \"\"\"GraphSAGE for node classification.\"\"\"

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.sage1 = SAGEConv(hidden_dim, aggregator="mean")
        self.sage2 = SAGEConv(num_classes, aggregator="mean")
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        # First SAGE layer
        x = self.sage1([x, edge_index])
        x = keras.ops.relu(x)
        x = self.dropout(x, training=training)

        # Second SAGE layer
        x = self.sage2([x, edge_index])
        return x

# Create and train SAGE model
print("\\nTraining GraphSAGE...")
sage_model = SAGENodeClassifier(
    input_dim=data.x.shape[1],
    hidden_dim=64,
    num_classes=len(np.unique(data.y))
)

sage_accuracy = train_node_classifier(sage_model, data)
```

## Deep GNN with Residual Connections

For deeper networks, residual connections help with training:

```python
class DeepGCN(keras.Model):
    \"\"\"Deep GCN with residual connections.\"\"\"

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=4, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = keras.layers.Dropout(dropout)

        # Input projection
        self.input_proj = keras.layers.Dense(hidden_dim)

        # GCN layers
        self.gcn_layers = [GCNConv(hidden_dim, use_bias=True) for _ in range(num_layers)]

        # Output layer
        self.output_layer = keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        # Input projection
        x = self.input_proj(x)
        x = keras.ops.relu(x)

        # GCN layers with residual connections
        for i, gcn in enumerate(self.gcn_layers):
            residual = x

            # GCN + activation
            x = gcn([x, edge_index])
            x = keras.ops.relu(x)
            x = self.dropout(x, training=training)

            # Residual connection
            x = x + residual

        # Output
        x = self.output_layer(x)
        return x

# Create and train deep model
print("\\nTraining Deep GCN...")
deep_model = DeepGCN(
    input_dim=data.x.shape[1],
    hidden_dim=64,
    num_classes=len(np.unique(data.y)),
    num_layers=4
)

deep_accuracy = train_node_classifier(deep_model, data, epochs=300)
```

## Model Comparison and Analysis

Let's compare the performance of different models:

```python
# Print comparison
results = {
    "GCN": gcn_accuracy,
    "GAT": gat_accuracy,
    "GraphSAGE": sage_accuracy,
    "Deep GCN": deep_accuracy
}

print("\\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for model_name, accuracy in results.items():
    print(f"{model_name:12s}: {accuracy:.4f}")

print("="*50)

# Best model
best_model = max(results.items(), key=lambda x: x[1])
print(f"Best model: {best_model[0]} ({best_model[1]:.4f})")
```

## Analyzing Node Embeddings

Let's visualize the learned node embeddings:

```python
def get_node_embeddings(model, x, edge_index):
    \"\"\"Extract node embeddings from trained model.\"\"\"
    # For GCN, get embeddings after first layer
    if hasattr(model, 'gcn1'):
        embeddings = model.gcn1([x, edge_index])
        embeddings = keras.ops.relu(embeddings)
    elif hasattr(model, 'gat1'):
        embeddings = model.gat1([x, edge_index])
        embeddings = keras.ops.elu(embeddings)
    elif hasattr(model, 'sage1'):
        embeddings = model.sage1([x, edge_index])
        embeddings = keras.ops.relu(embeddings)
    else:
        # For deep models, use intermediate layer
        x_proj = model.input_proj(x)
        embeddings = model.gcn_layers[0]([x_proj, edge_index])
        embeddings = keras.ops.relu(embeddings)

    return keras.ops.convert_to_numpy(embeddings)

# Get embeddings from best model
if best_model[0] == "GCN":
    best_embeddings = get_node_embeddings(model, data.x, data.edge_index)
elif best_model[0] == "GAT":
    best_embeddings = get_node_embeddings(gat_model, data.x, data.edge_index)
elif best_model[0] == "GraphSAGE":
    best_embeddings = get_node_embeddings(sage_model, data.x, data.edge_index)
else:
    best_embeddings = get_node_embeddings(deep_model, data.x, data.edge_index)

print(f"\\nNode embeddings shape: {best_embeddings.shape}")

# Simple analysis: compute embedding similarity within classes
from sklearn.metrics.pairwise import cosine_similarity

class_similarities = []
for class_id in np.unique(data.y):
    # Get embeddings for this class
    class_mask = data.y == class_id
    class_embeddings = best_embeddings[class_mask]

    # Compute pairwise similarities
    if len(class_embeddings) > 1:
        similarities = cosine_similarity(class_embeddings)
        # Get upper triangle (exclude diagonal)
        upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
        avg_similarity = np.mean(upper_tri)
        class_similarities.append(avg_similarity)

        print(f"Class {class_id}: avg intra-class similarity = {avg_similarity:.3f}")

print(f"\\nOverall avg intra-class similarity: {np.mean(class_similarities):.3f}")
```

## Best Practices for Node Classification

### 1. Data Preprocessing

```python
def preprocess_features(features):
    \"\"\"Common preprocessing steps for node features.\"\"\"
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # Remove features with low variance
    var_threshold = 0.01
    feature_var = np.var(features, axis=0)
    keep_features = feature_var > var_threshold
    features = features[:, keep_features]

    return features, keep_features

# Apply preprocessing
processed_features, kept_features = preprocess_features(data.x)
print(f"Features: {data.x.shape[1]} -> {processed_features.shape[1]}")
```

### 2. Model Selection Guidelines

- **GCN**: Good starting point, simple and effective
- **GAT**: When node importance varies significantly
- **GraphSAGE**: For large graphs or inductive settings
- **Deep models**: When you have sufficient data and computational resources

### 3. Hyperparameter Tuning

```python
def hyperparameter_search():
    \"\"\"Simple grid search for hyperparameters.\"\"\"

    configs = [
        {"hidden_dim": 32, "dropout": 0.3, "lr": 0.01},
        {"hidden_dim": 64, "dropout": 0.5, "lr": 0.01},
        {"hidden_dim": 128, "dropout": 0.7, "lr": 0.005},
        {"hidden_dim": 64, "dropout": 0.5, "lr": 0.02},
    ]

    best_config = None
    best_acc = 0

    for config in configs:
        print(f"\\nTesting config: {config}")

        # Create model with config
        test_model = SimpleGCN(
            input_dim=data.x.shape[1],
            hidden_dim=config["hidden_dim"],
            num_classes=len(np.unique(data.y)),
            dropout=config["dropout"]
        )

        # Train with config
        acc = train_node_classifier(
            test_model, data,
            epochs=100,  # Shorter for search
            lr=config["lr"]
        )

        if acc > best_acc:
            best_acc = acc
            best_config = config

    print(f"\\nBest config: {best_config} (acc: {best_acc:.4f})")
    return best_config

# Run hyperparameter search (optional - takes time)
# best_config = hyperparameter_search()
```

### 4. Evaluation Metrics

```python
def detailed_evaluation(model, data):
    \"\"\"Comprehensive evaluation with multiple metrics.\"\"\"
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    # Get predictions
    x = keras.ops.convert_to_tensor(data.x, dtype="float32")
    edge_index = keras.ops.convert_to_tensor(data.edge_index, dtype="int32")

    logits = model([x, edge_index], training=False)
    pred = keras.ops.convert_to_numpy(keras.ops.argmax(logits, axis=1))

    # Test set evaluation
    test_pred = pred[data.test_mask] if hasattr(data, 'test_mask') else pred[-200:]
    test_true = data.y[data.test_mask] if hasattr(data, 'test_mask') else data.y[-200:]

    # Compute metrics
    accuracy = accuracy_score(test_true, test_pred)
    f1_macro = f1_score(test_true, test_pred, average='macro')
    f1_micro = f1_score(test_true, test_pred, average='micro')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Micro): {f1_micro:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_true, test_pred)
    print(f"\\nConfusion Matrix:\\n{cm}")

    return accuracy, f1_macro, f1_micro

# Evaluate best model
print("\\nDetailed evaluation of best model:")
detailed_evaluation(model, data)  # Using the first GCN model as example
```

## Common Issues and Solutions

### Issue 1: Overfitting
```python
# Solution: Increase dropout, add regularization
model_regularized = SimpleGCN(
    input_dim=data.x.shape[1],
    hidden_dim=32,  # Smaller hidden dim
    num_classes=len(np.unique(data.y)),
    dropout=0.8  # Higher dropout
)
```

### Issue 2: Vanishing Gradients
```python
# Solution: Residual connections, batch normalization
class RegularizedGCN(keras.Model):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(hidden_dim, use_bias=True)
        self.bn1 = keras.layers.BatchNormalization()
        self.gcn2 = GCNConv(num_classes, use_bias=True)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        x = self.gcn1([x, edge_index])
        x = self.bn1(x, training=training)  # Batch norm
        x = keras.ops.relu(x)
        x = self.gcn2([x, edge_index])
        return x
```

### Issue 3: Class Imbalance
```python
# Solution: Weighted loss function
def compute_class_weights(labels):
    \"\"\"Compute class weights for imbalanced data.\"\"\"
    from collections import Counter

    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * count)

    return weights

# Usage in training loop
class_weights = compute_class_weights(data.y)
print(f"Class weights: {class_weights}")
```

## Conclusion

In this tutorial, you learned:

1. **Basic concepts** of node classification
2. **Multiple GNN architectures** (GCN, GAT, GraphSAGE)
3. **Training procedures** and evaluation metrics
4. **Advanced techniques** like residual connections
5. **Best practices** for real-world applications

### Next Steps

- Try different datasets (PubMed, CiteSeer)
- Experiment with ensemble methods
- Explore semi-supervised learning techniques
- Learn about inductive vs. transductive settings

### Further Reading

- [Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Veličković et al. (2018): Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Hamilton et al. (2017): Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

Happy node classifying! 🎯
