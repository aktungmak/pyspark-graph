from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, min as _min, least, explode, array, struct, greatest, count, sum as _sum, when
from pyspark.sql.types import DecimalType

from src.pyspark_graph.algorithms import Algorithm
from src.pyspark_graph.algorithms.pregel import Pregel
from src.pyspark_graph.graph import Graph, ID, SRC, DST

COMPONENT = "component"
ALGO_PREGEL = "pregel"
ALGO_ALTERNATING = "alternating"
ALGO_LAPLACIAN = "laplacian"


# TODO implement laplacian algorithm

class ConnectedComponents(Algorithm):
    """
    Identify connected components if the graph is undirected and strongly connected components
    if the graph is directed.

    :param max_iterations:
    """

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations

    def run(self, g) -> DataFrame:
        p = Pregel(initial_state=col(ID),
                   agg_expr=_min(Pregel.MSG),
                   msg_to_src=col(Pregel.STATE) if not g.directed else None,
                   msg_to_dst=col(Pregel.STATE),
                   update_expr=least(Pregel.MSG, Pregel.STATE),
                   max_iterations=self.max_iterations)
        return p.run(g).select(col(ID), col(Pregel.STATE).alias(COMPONENT))


class BSSSConnectedComponents(Algorithm):
    ORIG_ID = "orig_id"
    MIN_NBR = "min_nbr"
    CNT = "cnt"
    CHECKPOINT_NAME_PREFIX = "connected-components"

    def __init__(self, broadcastThreshold: int = 1000000, checkpointInterval: int = 2,
                 intermediateStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK):
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
        hubs = minNbrs.filter(col(self.CNT) > self.broadcastThreshold).select(
            col(SRC).cast("long")).distinct().collect()
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
            when(ee[SRC].isNull(), vv[ID]).otherwise(ee[SRC]).alias(COMPONENT))
