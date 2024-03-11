import typing
from typing import Optional

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, least, array, array_contains, array_append, min as _min, mode, sha1, array_join, lit, collect_list, max as _max, explode, struct, greatest, count, sum as _sum, \
    when
from pyspark.sql.types import StructType, StructField, LongType, ArrayType, DecimalType

from . import matrix
from .graph import Graph, ID, SRC, EDGE_ID, DST, DEGREE
from .matrix import graph_to_adjacency_matrix
from .util import order_edges, multiple_union, ne_null_safe


class Algorithm:
    pass


class BreadthFirstSearch(Algorithm):
    """
    Perform breadth-first search from a specified set of source vertices to a destination set of vertices,
    while optionally limiting the edges that can be traversed.
    The search is performed iteratively and is by default limited to 10 iterations.
    Returns start, end and arrays of the edges and vertices traversed.
    """
    START = "start"
    END = "end"
    EDGES = "edges"
    VERTICES = "vertices"
    result_schema = StructType([StructField(START, LongType(), False), StructField(END, LongType(), False),
                                StructField(EDGES, ArrayType(LongType(), False), False),
                                StructField(VERTICES, ArrayType(LongType(), False), False)])

    def __init__(self, start_expr: Column, end_expr: Column, edge_expr: Column = "true", max_iterations: int = 10):
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.edge_expr = edge_expr
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        if g.directed:
            edges = g.edges
        else:
            reverse = g.edges.withColumns({SRC: DST, DST: SRC})
            edges = g.edges.union(reverse).distinct()

        start = g.vertices.filter(self.start_expr)
        end = g.vertices.filter(self.end_expr)
        edges = edges.filter(self.edge_expr)

        # check for trivial empty result
        if start.isEmpty() or edges.isEmpty() or end.isEmpty():
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
    initial_state: the initial state of the vertex before iteration starts, can use all vertex columns
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
                 initial_state: Column,
                 agg_expr: Column,
                 msg_to_src: Column = None,
                 msg_to_dst: Column = None,
                 update_expr: Column = None,
                 comparison: typing.Callable = ne_null_safe,
                 max_iterations: int = 10,
                 checkpoint_interval: int = 2):
        if msg_to_src is None and msg_to_dst is None:
            raise ValueError("need at least one of msg_to_src or msg_to_dst")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than 0")
        self.initial_state = initial_state
        self.agg_expr = agg_expr
        self.msg_to_src = msg_to_src
        self.msg_to_dst = msg_to_dst
        self.update_expr = col(self.MSG) if update_expr is None else update_expr
        self.comparison = comparison
        self.max_iterations = max_iterations
        self.checkpoint_interval = checkpoint_interval

    def run(self, g: Graph) -> DataFrame:
        state = g.vertices.withColumns({self.STATE: self.initial_state,
                                        self.OLD_STATE: lit(None)})
        changed = state
        for i in range(self.max_iterations):
            # send msgs
            message_dfs = []
            if self.msg_to_src is not None:
                message_dfs.append(self._send(changed, g.edges, self.msg_to_src, DST, SRC))
            if self.msg_to_dst is not None:
                message_dfs.append(self._send(changed, g.edges, self.msg_to_dst, SRC, DST))
            messages = multiple_union(message_dfs)

            # aggregate messages
            agg_messages = messages.groupBy(ID).agg(self.agg_expr.alias(self.MSG))

            # update vertex state
            updated = agg_messages.join(state, ID) \
                .withColumns({self.OLD_STATE: col(self.STATE),
                              self.STATE: self.update_expr}) \
                .drop(self.MSG)
            # DataFrame does not support insert so we use antijoin+union
            not_updated = state.join(messages, ID, "anti")
            state = updated.union(not_updated)

            # check for termination
            changed = updated.filter(self.comparison(col(self.STATE), col(self.OLD_STATE)))
            if changed.isEmpty():
                break

            if g.checkpointing and self.checkpoint_interval > 0 and i % self.checkpoint_interval == 0:
                state = state.checkpoint()

        return state

    def _send(self,
              changed_vertices: DataFrame,
              edges: DataFrame,
              msg_expr: Column,
              _from: str,
              to: str) -> DataFrame:
        """Take a dataframe of changed vertices, apply the send expression
         and then join the result through the edge to the destination.
         Result is a DataFrame of recipients and messages."""
        return changed_vertices \
            .select(col(ID).alias(_from),
                    msg_expr.alias(self.MSG)) \
            .join(edges, _from) \
            .select(col(to).alias(ID), col(self.MSG))


# join changed vertices


class ConnectedComponents(Algorithm):
    """
    Identify connected components if the graph is undirected and strongly connected components
    if the graph is directed.

    :param max_iterations:
    """
    COMPONENT = "component"
    ALGO_PREGEL = "pregel"
    ALGO_ALTERNATING = "alternating"
    ALGO_LAPLACIAN = "laplacian"

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations

    def run(self, g: Graph, algo=ALGO_PREGEL):
        if algo == self.ALGO_PREGEL:
            return self._run_pregel(g)
        elif algo == self.ALGO_ALTERNATING:
            return self._run_alternating(g)
        else:
            raise ValueError(f"unknown connected components algorith {algo}")

    def _run_pregel(self, g) -> DataFrame:
        p = Pregel(initial_state=col(ID),
                   agg_expr=_min(Pregel.MSG),
                   msg_to_src=col(Pregel.STATE) if not g.directed else None,
                   msg_to_dst=col(Pregel.STATE),
                   update_expr=least(Pregel.MSG, Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.select(col(ID), col(Pregel.STATE).alias(self.COMPONENT))

    def _run_alternating(self, g) -> DataFrame:
        # normalise graph
        ordered = order_edges(g)
        # big star - connect all neighbours > v to smallest neighbour
        nmin = self.neighbourhood_min(ordered)
        bigstar = g.edges.join(nmin, SRC).select()

        # small star - connect all neighbours <= v to smallest neighbour
        # check for convergence - sum/mean-median all component assignments
        raise NotImplementedError()

    def _neighbourhood_min(self, edges: DataFrame) -> DataFrame:
        "minimum ID in each node's neighbourhood (including itself)"
        return edges.groupBy(SRC) \
            .agg(_min(DST).alias("min")) \
            .select(col(SRC).alias(ID), least(SRC, "min"))

    def _run_laplacian(self, g) -> DataFrame:
        raise NotImplementedError()


class LabelPropagation(Algorithm):
    """
    Starting with an initial label (by default the vertex ID if a label column is not
    provided) each vertex updates its label based on the modal value of its neighbours.
    This is repeated for a fixed number of iterations but may converge earlier.

    :param label_column: A Column expression defining the initial label. If not specified the
                         vertex ID is used instead.
    :param max_iterations: By default 10, may be less if the graph stage converges earler.
    """
    LABEL = "label"

    def __init__(self, label_column: Column = None, max_iterations=10):
        self.label_column = label_column
        self.max_iterations = max_iterations

    def run(self, g: Graph):
        p = Pregel(initial_state=col(ID) if self.label_column is None else self.label_column,
                   agg_expr=mode(Pregel.MSG),
                   msg_to_src=None if g.directed else col(Pregel.STATE),
                   msg_to_dst=col(Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.select(col(ID), col(Pregel.STATE).alias(self.LABEL))


class WLKernel(Algorithm):
    """Calculate the Weisfeiler-Lehman kernel for the graph
    Returns a hash that will be the same for two isomorphic graphs.
    By default, the degree of the node will be used as the starting label,
    but any vertex column can be used instead."""

    def __init__(self, hashfunc=sha1, label_column: str = None, max_iterations=3):
        self.hash = hashfunc
        self.label_column = label_column
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> str:
        label = self.label_column
        if label is None:
            vertices = g.vertices.join(g.degrees, ID)
            # TODO find a good way to update the graph
            g = Graph(vertices, g.edges, indexed=True)
            label = DEGREE
        p = Pregel(initial_state=col(label),
                   agg_expr=self.hash(array_join(collect_list(Pregel.MSG), "")),
                   msg_to_src=None if g.directed else col(Pregel.STATE),
                   msg_to_dst=col(Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.agg(self.hash(array_join(collect_list(Pregel.STATE), ""))).first()[0]


class KatzIndex(Algorithm):
    """
    Calculate the Katz index of each pair of vertices, which represents the number
    of paths up to max_iterations long there are between pairs of vertices.

    """
    INDEX = "katz_index"

    def __init__(self, beta: float = 1.0, tolerance: float = None, max_iterations: int = 10):
        self.beta = beta
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        a = graph_to_adjacency_matrix(g)
        for _ in range(self.max_iterations):
            a *= a
            a.df.show()
            if self.tolerance is not None:
                # check if average difference is below tolerance
                max_delta = a.df.agg(_max(matrix.VAL)).first()[0]
                if max_delta < self.tolerance:
                    break

        return a.df.withColumnsRenamed({matrix.ROW: SRC,
                                        matrix.COL: DST,
                                        matrix.VAL: self.INDEX})


class BSSSConnectedComponents(Algorithm):
    COMPONENT = "component"
    ORIG_ID = "orig_id"
    MIN_NBR = "min_nbr"
    CNT = "cnt"
    CHECKPOINT_NAME_PREFIX = "connected-components"

    def __init__(self, broadcastThreshold: int = 1000000, checkpointInterval: int = 2, intermediateStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
        self.broadcastThreshold = broadcastThreshold
        self.checkpointInterval = checkpointInterval
        self.intermediateStorageLevel = intermediateStorageLevel

    def symmetrize(self, ee):
        """Add in extra edges to make the directed graph symmetric (i.e. undirected)"""
        EDGE = "_edge"
        return (ee.select(explode(array(
            struct(col("src"), col("dst")),
            struct(col("dst").alias("src"), col("src").alias("dst")))).alias(EDGE))
                .select(col(f"{EDGE}.src").alias("src"), col(f"{EDGE}.dst").alias("dst")))

    def prepare(self, g: Graph):
        vertices = g.vertices.select(col(ID)).distinct()
        edges = g.edges.select(col(SRC), col(DST))
        orderedEdges = edges.filter(col(SRC) != col(DST)).select(least(col(SRC), col(DST)).alias(SRC),
                                                                 greatest(col(SRC), col(DST)).alias(DST)).distinct()
        return Graph(vertices, orderedEdges)

    def minNbrs(self, edges, includeSelf: bool):
        # TODO merge branches below and just add lesser() if includeSelf
        if includeSelf:
            # TODO do we need to symmetrise each iteration?
            return (self.symmetrize(edges)
                    .groupBy(SRC)
                    .agg(_min(col(DST)).alias(self.MIN_NBR),
                         count("*").alias(self.CNT))
                    .withColumn(self.MIN_NBR, least(col(SRC), col(self.MIN_NBR))))
        else:
            # TODO do we need to symmetrise here as well?
            return (edges
                    .groupBy(SRC)
                    .agg(_min(col(DST)).alias(self.MIN_NBR),
                         count("*").alias(self.CNT)))

    def skewedJoin(self, edges: DataFrame, minNbrs: DataFrame):
        hubs = minNbrs.filter(col(self.CNT) > self.broadcastThreshold).select(col(SRC).cast("long")).distinct().collect()
        return Graph.skewedJoin(edges, minNbrs, SRC, hubs, logPrefix)

    def run(self, graph: Graph):
        runId = "todo uuid"
        logPrefix = f"[CC {runId}]"
        print(f"{logPrefix} Start connected components with run ID {runId}.")

        sc = graph.spark.sparkContext

        shouldCheckpoint = self.checkpointInterval > 0
        if shouldCheckpoint:
            checkpointDir = sc.getCheckpointDir()
            if checkpointDir is None:
                raise Exception("Checkpoint directory is not set. Please set it first using sc.setCheckpointDir().")
            else:
                checkpointDir += f"{self.CHECKPOINT_NAME_PREFIX}-{runId}"
                print(f"{logPrefix} Using {checkpointDir} for checkpointing with interval {self.checkpointInterval}.")
        else:
            print(f"{logPrefix} Checkpointing is disabled because checkpoint interval is {self.checkpointInterval}.")

        print(f"{logPrefix} Preparing the graph for connected component computation ...")
        g = self.prepare(graph)
        vv = g.vertices
        ee = g.edges.persist(self.intermediateStorageLevel)
        numEdges = ee.count()
        print(f"{logPrefix} Found {numEdges} edges after preparation.")

        converged = False
        iteration = 1
        prevSum = None
        while not converged:
            # large-star step #############################################################
            # compute min node id in neighborhood (including self)
            minNbrs1 = self.minNbrs(ee, includeSelf=True).persist(self.intermediateStorageLevel)

            # connect all strictly larger neighbors to the min neighbor (including self)
            ee = self.skewedJoin(ee, minNbrs1).select(col(DST).alias(SRC),
                                                                                          col(self.MIN_NBR).alias(
                                                                                              DST)).distinct().persist(
                self.intermediateStorageLevel)

            # small-star step #############################################################
            # compute min node id in neighborhood (excluding self)
            minNbrs2 = self.minNbrs(ee, includeSelf=False).persist(self.intermediateStorageLevel)

            # connect all smaller neighbors to the min neighbor
            ee = self.skewedJoin(ee, minNbrs2, self.broadcastThreshold, logPrefix).select(col(self.MIN_NBR).alias(SRC),
                                                                                          col(DST)).filter(
                col(SRC) != col(DST))
            # connect self to the min neighbor
            ee = ee.union(minNbrs2.select(col(self.MIN_NBR).alias(SRC), col(SRC).alias(DST))).distinct()

            # checkpointing
            if shouldCheckpoint and (iteration % self.checkpointInterval == 0):
                # TODO check whether checkpoint is equivalent to prev parquet method
                ee.checkpoint()

            ee.persist(self.intermediateStorageLevel)

            # test convergence
            # TODO find a better way to detect convergence
            currSum, cnt = ee.select(_sum(col(SRC).cast(DecimalType(20, 0))), count("*")).first().collect()
            if cnt != 0 and currSum is None:
                raise ArithmeticError(f"sum of edge IDs overflowed")

            print(f"{logPrefix} Sum of assigned components in iteration $iteration: {currSum}.")
            if currSum == prevSum:
                # This also covers the case when cnt = 0 and currSum is null, which means no edges.
                converged = True
            else:
                prevSum = currSum

            iteration += 1

        print(f"{logPrefix} Connected components converged in {iteration - 1} iterations.")

        return vv.join(ee, vv[ID] == ee[DST], "left_outer").select(
            when(ee[SRC].isNull(), vv[ID]).otherwise(ee[SRC]).alias(self.COMPONENT))
