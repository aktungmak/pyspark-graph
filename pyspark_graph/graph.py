from functools import cached_property
from typing import Optional

import pyspark.sql
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

SRC = "src"
DST = "dst"
ID = "id"
EDGE_ID = "edge_id"
ADJ = "adjacent"
OLD_SRC = "old_src"
OLD_DST = "old_dst"
OLD_ID = "old_id"
DEGREE = "degree"
IN_DEGREE = "in_degree"
OUT_DEGREE = "out_degree"

class Graph:
    def __init__(self,
                 vertices: DataFrame, edges: DataFrame,
                 directed: bool = True,
                 indexed: bool = False,
                 spark: Optional[SparkSession] = None):
        f"""
        Create a new Graph from vertices and edges or an adjacency list 
        :param vertices: DataFrame with a column "{ID}" plus any user attributes
        :param edges: Dataframe with columns "{SRC}" and "{DST}" plus any user attributes
        :param directed: Should the graph be treated as if it is directed?
        :param indexed: Does the graph already have distinct edges indexed with LONG keys?
        """
        self._v = vertices
        self._e = edges
        self._directed = directed
        self.spark = spark if spark is not None else SparkSession.getActiveSession()
        self._checkpointing = self.spark.sparkContext.getCheckpointDir() is not None

        if not indexed:
            self._index()

    def _index(self):
        """
        Assign LONG ids to all vertices and edges in the graph to improve performance
        """
        if OLD_ID in self._v.columns:
            raise ValueError(f"vertices dataframe already contains a column {OLD_ID}")
        if OLD_SRC in self._e.columns:
            raise ValueError(f"edges dataframe already contains a column {OLD_SRC}")
        if OLD_DST in self._e.columns:
            raise ValueError(f"edges dataframe already contains a column {OLD_DST}")

        # TODO repartition and/or sort within partitions?
        v = self._v.distinct() \
            .withColumnRenamed(ID, OLD_ID) \
            .withColumn(ID, F.monotonically_increasing_id())

        e = self._e.distinct() \
            .withColumnsRenamed({SRC: OLD_SRC, DST: OLD_DST})
        e = e.join(v.withColumnRenamed(ID, SRC), e[OLD_SRC] == v[OLD_ID]) \
            .drop(OLD_ID) \
            .join(v.withColumnRenamed(ID, DST), e[OLD_DST] == v[OLD_ID]) \
            .select(F.monotonically_increasing_id().alias(EDGE_ID), SRC, DST, e["*"])

        if self._checkpointing:
            v = v.checkpoint()
            e = e.checkpoint()

        self._v = v
        self._e = e

    # TODO add options to persist graph
    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def vertices(self) -> DataFrame:
        return self._v

    @property
    def edges(self) -> DataFrame:
        return self._e

    @cached_property
    def adjacency(self) -> DataFrame:
        connected = self._e.select(F.col(SRC), F.col(DST))
        if not self._directed:
            reverse_edges = self._e.select(F.col(SRC).alias(DST), F.col(DST).alias(SRC))
            connected = connected.union(reverse_edges)
        grouped = connected.groupBy(F.col(SRC).alias(ID)).agg(F.collect_list(DST).alias(ADJ))
        isolated = self._v.select(F.col(ID), F.array().alias(ADJ)).join(grouped, ID, "anti")
        adjacency = grouped.union(isolated)
        if self._checkpointing():
            adjacency = adjacency.checkpoint()
        return adjacency

    @property
    def out_degrees(self) -> DataFrame:
        return self._e.groupBy(F.col(SRC).alias(ID)).agg(F.count("*").alias(OUT_DEGREE))

    @property
    def in_degrees(self) -> DataFrame:
        return self._e.groupBy(F.col(DST).alias(ID)).agg(F.count("*").alias(IN_DEGREE))

    @property
    def degrees(self) -> DataFrame:
        if self._directed:
            return self.out_degrees.withColumnRenamed(OUT_DEGREE, DEGREE)
        else:
            return self.adjacency.select(F.col(ID), F.size(F.col(ADJ)).alias(DEGREE))

    def triplets(self, src_vertex_prefix: str, dst_vertex_prefix: str) -> DataFrame:
        src_vertices = self._v.toDF(*[src_vertex_prefix + c for c in g.vertices.columns])
        dst_vertices = self._v.toDF(*[dst_vertex_prefix + c for c in g.vertices.columns])
        return self._e \
            .join(src_vertices, self._e[SRC] == src_vertices[src_vertex_prefix + ID]) \
            .join(dst_vertices, self._e[DST] == dst_vertices[dst_vertex_prefix + ID])

    @property
    def checkpointing(self):
        return self._checkpointing
