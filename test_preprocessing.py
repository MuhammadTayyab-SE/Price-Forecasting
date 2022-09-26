import unittest
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Row
import math
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer


class MyTestCase(unittest.TestCase):
    agg_df = None
    def test_aggregation_transformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()
        dataframe = spark.read.csv('test_dataset/testdata.csv', inferSchema=True, header=True)
        transformer = AggregatedTransformer()
        agg_df = transformer.transform(dataframe)
        with self.subTest():
            self.assertIn('sales', agg_df.columns)
        with self.subTest():
            self.assertEqual(agg_df.count(), dataframe.count())


    def test_mark_zero_neg_transformer(self):
        columns = ["sales"]
        data = [20000, 100000, 3000, 0]
        spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
        rdd = spark.sparkContext.parallelize(data)
        dfff = rdd.map(lambda x: (x,)).toDF(columns)

        transformer = MarkZeroNegTransformer()
        agg_df = transformer.transform(dfff)

        # assert 'flag' in agg_df.columns
        self.assertIn('flag', agg_df.columns)

    def test_impute_mean_transformer(self):

        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()
        dataframe = spark.read.csv('testdata/testdata.csv', inferSchema=True, header=True)
        transformer = ImputeMeanTransformer()
        agg_df = transformer.transform(dataframe)
        with self.subTest():
            self.assertEqual(len(agg_df.select('store_id').collect()), 2)



    def test_train_test_transformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()
        dataframe = spark.read.csv('test_dataset/testdata.csv', inferSchema=True, header=True)
        transformer = TrainTestTransformer()
        train, test = transformer.transform(dataframe)
        print(train.count(), test.count())
        with self.subTest():
            self.assertEqual(23, train.count())
        with self.subTest():
            self.assertEqual(9, test.count())


if __name__ == '__main__':
    unittest.main()
