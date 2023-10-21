# GraphTLP
Transductive Label Propagation Algorithms

# Resources
- [Datasets](https://github.com/BrucceNeves/TSRN4HEN/tree/master/datasets) and [Labels](https://github.com/BrucceNeves/TSRN4HEN/tree/master/labels): twelve heterogeneous event networks from different domains (extracted from Reuters Corpus).

# Available Algorithms
- Homogeneous Networks:
  * GFHF [(Zhu et al., 2003)](https://research.cs.wisc.edu/bullying/pub/zgl.pdf)
  * LLGC [(Zhou et al., 2004)](https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html)
- Heterogeneous Networks:
  * GNetMine [(Ji et al., 2010)](https://link.springer.com/chapter/10.1007/978-3-642-15880-3_42)
  * LPHN [(Rossi et al., 2014)](https://dl.acm.org/doi/abs/10.1145/2554850.2554901)

# Instalation
> pip install git+git://github.com/BrucceNeves/GraphTLP

# How to Use
```python
import networkx as nx, pandas as pd, numpy as np
from graphtlp.network import tlp_network
from graphtlp import GFHF, LLGC, GNetMine, LPHN
```

# 1) Graph Creation
For convenience, we will generate a random bipartite graph using networkx, however it works with any type of undirected graph. In addition, we will put a weight of 1 for all edges, but it can be any value > 0 and <= 1.
```python
G = bipartite.random_graph(10, 40, 0.5)
for _, _, d in G.edges(data=True):
  d['weight'] = 1
```

It is necessary that all the vertices are associated with a layer, for that we will rename all the vertices to the "node_name:layer" format, where "layer" indicates which layer these vertices belong to.

For heterogeneous networks, we will use the information available in the vertex, however it is possible to use other names.
```python
nx.relabel_nodes(G, {n: f'{n}:layer_{d["bipartite"]}' for n, d in G.nodes(data=True)}, copy=False)
```
If you want to create a homogeneous network, use a default name for all vertices.
```python
nx.relabel_nodes(G, {n: f'{n}:layer' for n in G.nodes()}, copy=False)
```
That's all you need to create a network from scratch.

## 1.1) Alternatively:
You can also go to the link we provide in Resources section and download one of the datasets that is available there, extract the contents of the zip and load the graph dataset using "read_weighted_edgelist" from networkx. Considering that you downloaded the graph dataset "business_transactions.zip" and extracted it, the load would look like this:
```python
G = nx.read_weighted_edgelist("business_transactions.edges", delimiter='\t')
```

# 2) Labels
The expected format for the labels is a pandas DataFrame, in which each line refers to a labeled vertex and its class, the class is represented by a vector in the one hot format, with the value 1 in the position referring to the label of that node.

For example below we are defining that nodes '0:layer_0' and '9:layer_0' are labeled nodes. We also defined the problem as binary classification, so its classes are 'A' and 'B' respectively, which were mapped to [1, 0] for class A, while class B became [0, 1].
```python
labels = pd.DataFrame([{'node':'0:layer_0', 'y': np.array([1, 0])}, {'node': '9:layer_0', 'y': np.array([0, 1])}]).set_index('node')
```
## 2.1) Alternatively:
You can also go to the link we provide in Resources section and download one of the datasets that is available there, extract the contents of the zip and load the labels using "read_csv" from pandas. Considering that you downloaded the graph dataset "business_transactions.zip" and extracted it, the load would look like this:
```python
labels = pd.read_csv('business_transactions.full_labels', sep='\t', names=['node', 'y']).set_index('node')
labels['y'] = labels['y'].map(lambda x: np.array([float(y) for y in x.split(',')]))
```
As the label values are stored as a string in this dataset, it had to be converted to a vector containing floats.

# 3) Label Propagation
To use the algorithms it is necessary to instantiate an object of type "tlp_network", for that we do:
```python
network = tlp_network(G, labels)
```
Before instantiating the algorithms, it is necessary to define the parameters, these parameters are described below:
- **mi:** Importance of labeled data during the propagation of labels, ranging from 0.1 to 1. Used in **LLGC** and **GNetMine**
- **weight_relations:** Weight of the relations between the layers, the names of the layers must be connected by '|' and the values will be automatically normalized when running GNetMine, also all existing layer relations must be defined. If no pair of layers is specified all pairs of layers will have equal weights. Used only in **GNetMine**
```python
mi = 0.9
weight_relations = {}
```
As our example network only has 2 layers and is a bipartite graph, then we only have 1 type of relationship, which is between layer_0 and layer_1, but if there were more types of relationship, for example between layer_0 and layer_0 or layer_1 and layer_1, then it would be possible to give a different weight to each type of relationship, the code would look like this:
```python
weight_relations = {'layer_0|layer_1': 1, 'layer_0|layer_0': 0.5, 'layer_1|layer_1': 2}
```
The code below allows checking which were the relations created after the construction of the network.
```python
network.edges['r_type'].unique()
```
There are two more parameters that are related to the training of algorithms, they are:
- **iterations:** Maximum number of iterations, if the algorithm takes time to converge
- **convergenceThreshold:** Threshold to consider that the network has converged
```python
iterations = 1000
convergenceThreshold = 0.00005
```

Next, let's instantiate the algorithms:
```python
total_iterations, output = GFHF().train(iterations, convergenceThreshold, network)
```
```python
total_iterations, output = LLGC(mi).train(iterations, convergenceThreshold, network)
```
```python
total_iterations, output = GNetMine(mi, weight_relations).train(iterations, convergenceThreshold, network)
```
```python
total_iterations, output = LPHN().train(iterations, convergenceThreshold, network)
```

# 4) Evaluation
Section under construction
