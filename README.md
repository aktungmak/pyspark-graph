## pyspark-graph

This is a pure pyspark implementation of graph algorithms.
Many of these capabilites are already available in GraphX and GraphFrames,
but the language choice limits accessiblity for those who are not 
familiar with Scala.

Additionally, those libraries offer just the basic tools needed to implement
graph analytics whereas here we aim to offer a more batteries-included approach.

### Installation
This package is available on PyPI, to install it simply run:
```
pip install pyspark-graph
```

### Supported algorithms
The following table compares the features of pyspark-graph with GraphFrames and GraphX. The goal is to add the missing features and continue to add additional algorithms in future.

| Name                         | GraphX | GraphFrames | pyspark-graph |
|------------------------------|--------|-------------|---------------|
| AggregateMessages            | ✅      | ✅           | ✅             |
| BFS                          | ✅      | ✅           | ✅             |
| ConnectedComponents          | ✅      | ✅           | ✅             |
| LabelPropagation             | ✅      | ❌           | ✅             |
| PageRank                     | ✅      | ❌           | ❌             |
| ParallelPersonalizedPageRank | ✅      | ❌           | ❌             |
| Pregel                       | ✅      | ✅           | ✅             |
| SVDPlusPlus                  | ✅      | ❌           | ❌             |
| ShortestPaths                | ✅      | ❌           | ❌             |
| StronglyConnectedComponents  | ✅      | ❌           | ❌             |
| TriangleCount                | ✅      | ✅           | ✅             |
| JaccardSimilarity            | ❌      | ❌           | ✅             |
| OverlapCoefficient           | ❌      | ❌           | ✅             |