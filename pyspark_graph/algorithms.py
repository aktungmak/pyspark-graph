import operator
import typing
from typing import Optional

from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, array_intersect, array_union, least, array, array_contains, array_append, size, \
    lit
from pyspark.sql.types import StructType, StructField, LongType, ArrayType

from graph import Graph, ID, ADJ, SRC, EDGE_ID, DST
from pyspark_graph import util
from util import match_structure, order_edges, multiple_union


class Algorithm:
    pass


class TriangleCount(Algorithm):
    def run(self, g: Graph):
        return match_structure(order_edges(g.edges),
                               [("a", "b"), ("b", "c"), ("a", "c")]).count()


class JaccardSimilarity(Algorithm):
    name = "jaccard_similarity"

    def run(self, g: Graph) -> DataFrame:
        return g.adjacency.alias("a") \
            .join(g.adjacency.alias("b"),
                  col(f"a.{ID}") != col(f"b.{ID}")) \
            .select(f"a.{ID}", f"b.{ID}",
                    (size(array_intersect(f"a.{ADJ}", f"b.{ADJ}")) /
                     size(array_union(f"a.{ADJ}", f"b.{ADJ}")))
                    .alias(self.name))


class OverlapCoefficient(Algorithm):
    name = "overlap_coefficient"

    def run(self, g: Graph) -> DataFrame:
        return g.adjacency.alias("a") \
            .join(g.adjacency.alias("b"),
                  col(f"a.{ID}") != col(f"b.{ID}")) \
            .select(f"a.{ID}", f"b.{ID}",
                    (size(array_intersect(f"a.{ADJ}", f"b.{ADJ}")) /
                     least(size(f"a.{ADJ}"), size(f"b.{ADJ}")))
                    .alias(self.name))


class BreadthFirstSearch(Algorithm):
    """
    Perform breadth-first search from a specified set of source vertices to a destination set of vertices,
    while optionally limiting the edges that can be traversed.
    The search is performed iteratively and is by default limited to 100 iterations.
    Returns start, end and arrays of the edges and vertices traversed.
    """
    START = "start"
    END = "end"
    EDGES = "edges"
    VERTICES = "vertices"
    result_schema = StructType([StructField(START, LongType(), False), StructField(END, LongType(), False),
                                StructField(EDGES, ArrayType(LongType(), False), False),
                                StructField(VERTICES, ArrayType(LongType(), False), False)])

    def __init__(self, start_expr: Column, end_expr: Column, edge_expr: Column = "true", max_iterations: int = 100):
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.edge_expr = edge_expr
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        if not g.directed:
            # TODO undirected version
            raise NotImplementedError("BFS only implemented on directed graphs")
        start = g.vertices.filter(self.start_expr)
        end = g.vertices.filter(self.end_expr)
        edges = g.edges.filter(self.edge_expr)
        # check for trivial empty result
        if start.isEmpty() or edges.isEmpty() or end.isEmpty():
            print("no match for start, end or edges expressions")
            return g.spark.createDataFrame([], self.result_schema)

        horizon = "horizon"
        paths = start.select(col(ID).alias(self.START),
                             col(ID).alias(horizon),
                             array().alias(self.EDGES),
                             array(col(ID)).alias(self.VERTICES))
        for _ in range(self.max_iterations):
            # check if we reached end or ran out of paths
            result = paths.join(end, paths[horizon] == end[ID])
            if result.head(1) or paths.isEmpty():
                break
            # otherwise extend horizon
            paths = paths.join(edges, (edges[SRC] == paths[horizon]) \
                               & ~array_contains(paths[self.EDGES], edges[EDGE_ID])) \
                .select(col(self.START),
                        col(DST).alias(horizon),
                        array_append(col(self.EDGES), col(EDGE_ID)).alias(self.EDGES),
                        array_append(col(self.VERTICES), col(DST)).alias(self.VERTICES))
        else:
            print("max_iterations reached")
            return g.spark.createDataFrame([], self.result_schema)
        return result.select(col(self.START),
                             col(ID).alias(self.END),
                             col(self.EDGES),
                             col(self.VERTICES))


class AggregateMessages(Algorithm):
    """
    Aggregate messages or something.
    Maybe remove given poor iterative performance in Spark
    """
    MSG = "message"
    SRC_VERTEX_PREFIX = "src_vertex_"
    DST_VERTEX_PREFIX = "dst_vertex_"

    # result_schema = StructType([StructField(ID, LongType, False), StructField(MSG, Any, False)])

    def __init__(self, agg: Column, to_src: Optional[Column] = None, to_dst: Optional[Column] = None):
        if to_src is None and to_dst is None:
            raise ValueError("need at least one of to_src or to_dst")
        self.agg = agg
        self.to_src = to_src
        self.to_dst = to_dst

    def src_col(self, col_name: str) -> Column:
        return col(self.SRC_VERTEX_PREFIX + col_name)

    def dst_col(self, col_name: str) -> Column:
        return col(self.DST_VERTEX_PREFIX + col_name)

    def run(self, g: Graph) -> DataFrame:
        triplets = g.triplets(self.SRC_VERTEX_PREFIX, self.DST_VERTEX_PREFIX)
        if self.to_src and self.to_dst:
            to_src = triplets.select(self.to_src.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
            to_dst = triplets.select(self.to_dst.alias(self.MSG), col(self.DST_VERTEX_PREFIX + ID).alias(ID))
            result = to_src.union(to_dst)
        elif self.to_src:
            result = triplets.select(self.to_src.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
        elif self.to_dst:
            result = triplets.select(self.to_dst.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
        return result.groupBy(ID).agg(self.agg)


class Pregel(Algorithm):
    """
    initial_msg: will be sent to all vertices before the iteration starts, can use all vertex columns
    update_expr: executed at the start of each iteration to update state, can use all vertex columns plus MSG
    agg_expr: an expression on Pregel.MSG to aggregate all messages arriving at a vertex
    changed: a function that takes the old and new state and returns True if the state changed
    msg_to_dst: will be sent to all out-neighbours after state update, can use all vertex columns
    msg_to_dst: will be sent to all in-neighbours after state update, can use all vertex columns
    max_iterations: limit number of supersteps if no convergence
    """
    STATE = "state"
    OLD_STATE = "old_state"
    MSG = "message"  # column containing aggregated messages

    def __init__(self,
                 initial_msg: Column,
                 update_expr: Column,
                 agg_expr: Column,
                 changed: typing.Callable = operator.ne,
                 msg_to_src: Column = None,
                 msg_to_dst: Column = None,
                 max_iterations: int = 100,
                 checkpoint_interval: int = 5):
        if not (msg_to_src or msg_to_dst):
            raise ValueError("need at least one of msg_to_src or msg_to_dst")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than 0")
        self.initial_msg = initial_msg
        self.update_expr = update_expr
        self.agg_expr = agg_expr
        self.comparison = changed
        self.msg_to_src = msg_to_src
        self.msg_to_dst = msg_to_dst
        self.max_iterations = max_iterations
        self.checkpoint_interval = checkpoint_interval

    def run(self, g: Graph) -> DataFrame:
        v = g.vertices.select(self.initial_msg.alias(self.MSG),
                              lit(None).alias(self.STATE),
                              lit(None).alias(self.OLD_STATE))
        for i in range(self.max_iterations):
            # update vertex state
            updated = v.filter(col(self.MSG).isNotNull) \
                .withColumnRenamed(self.STATE, self.OLD_STATE) \
                .withColumn(self.STATE, self.update_expr)

            # merge updated state back in to v
            not_updated = v.join(updated, ID, "anti")
            v = updated.union(not_updated)

            # check for termination
            changed = updated.filter(self.comparison(col(self.STATE), col(self.OLD_STATE)))
            if changed.isEmpty():
                break

            # send msgs
            # TODO send to src
            messages = []
            if self.msg_to_src:
                messages.append(self._send(changed, g.edges, v, self.msg_to_src))
            if self.msg_to_dst:
                messages.append(self._send(changed, g.edges, v, self.msg_to_dst))
            messages = multiple_union(messages)

            # aggregate messages
            agg_messages = messages.groupBy(messages[ID]).agg(self.agg_expr.alias(self.MSG))

            # left outer join original vs with messages, leaving nulls where there are no messages
            v = v.drop(self.MSG).join(agg_messages, ID, "left")

            if self.checkpoint_interval > 0 and i % self.checkpoint_interval == 0:
                v = v.checkpoint()
        else:
            print("max_iterations reached")
        print(f"pregel terminated after {i} iterations")
        return v

    def _send(self, changed_vertices: DataFrame, edges: DataFrame, all_vertices: DataFrame, msg_expr: Column):
        if msg_expr is None:
            return None
        return changed_vertices \
            .select(col(ID).alias(SRC),
                    msg_expr.alias(self.MSG)) \
            .join(edges) \
            .join(all_vertices, all_vertices[ID] == edges[DST]) \
            .select(col(DST).alias(ID), col(self.MSG))
