from tests import samples
from tests.spark_test import SparkTest


class TestKatzIndex(SparkTest):
    def test_run(self):
        g = samples.sample1(self.spark)
        k = KatzIndex.KatzIndex(max_iterations=2)
        k.run(g).show()

        # {0: 0.43046326669401147,
        #  1: 0.5087293033711362,
        #  4: 0.43046326669401147,
        #  2: 0.43046326669401147,
        #  3: 0.43046326669401147}
