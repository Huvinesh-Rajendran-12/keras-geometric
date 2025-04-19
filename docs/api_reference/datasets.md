# Datasets

## Base Classes

### Dataset

```python
class Dataset
```

Base class for graph datasets in Keras Geometric.

**Arguments:**

- **root** (*str*): Root directory where the dataset should be saved
- **name** (*str*): Name of the dataset
- **transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version
- **pre_transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version, applied before the dataset is saved to disk

**Methods:**

- **split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True, seed=None)**: Split the dataset into training, validation, and test sets
- **__len__()**: Return the number of graphs in the dataset
- **__getitem__(idx)**: Return the graph at index idx

## Citation Datasets

### CitationDataset

```python
class CitationDataset(Dataset)
```

Base class for citation network datasets like Cora, CiteSeer, and PubMed.

**Arguments:**

- **root** (*str*): Root directory where the dataset should be saved
- **name** (*str*): Name of the dataset ('cora', 'citeseer', 'pubmed')
- **transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version
- **pre_transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version, applied before the dataset is saved to disk

### Cora

```python
class Cora(CitationDataset)
```

The Cora citation network dataset. Nodes represent scientific publications and edges represent citations.

**Arguments:**

- **root** (*str*): Root directory where the dataset should be saved. Default is "data".
- **transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version
- **pre_transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version, applied before the dataset is saved to disk

**Stats:**

- 2708 nodes
- 5429 edges
- 7 classes
- 1433 features per node

### CiteSeer

```python
class CiteSeer(CitationDataset)
```

The CiteSeer citation network dataset. Nodes represent scientific publications and edges represent citations.

**Arguments:**

- **root** (*str*): Root directory where the dataset should be saved. Default is "data".
- **transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version
- **pre_transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version, applied before the dataset is saved to disk

**Stats:**

- 3327 nodes
- 4732 edges
- 6 classes
- 3703 features per node

### PubMed

```python
class PubMed(CitationDataset)
```

The PubMed citation network dataset. Nodes represent scientific publications and edges represent citations.

**Arguments:**

- **root** (*str*): Root directory where the dataset should be saved. Default is "data".
- **transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version
- **pre_transform** (*callable, optional*): A function/transform that takes in a GraphData object and returns a transformed version, applied before the dataset is saved to disk

**Stats:**

- 19717 nodes
- 44338 edges
- 3 classes
- 500 features per node
