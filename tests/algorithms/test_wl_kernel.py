from tests import samples
from tests.spark_test import SparkTest


class TestWLKernel(SparkTest):

    def test_labelled(self):
        g = samples.labelled(self.spark)
        r = WLKernel.WLKernel(label_column="label").run(g)
        self.assertEqual('874ee76d142f4da4b531b6c5543b292278027430', r)

    def test_unlabelled(self):
        g = samples.labelled(self.spark)
        r = WLKernel.WLKernel().run(g)
        self.assertEqual('766dd9dcdfc8de175410960d32ad84e8a6116c4e', r)
