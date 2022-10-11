import unittest
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
import math
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer


class MyTestCase(unittest.TestCase):
    agg_df = None

    def test_aggregation_transformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()

        # reading the custom dataframe
        dataframe = spark.read.csv('test_dataset/testdata.csv', inferSchema=True, header=True)
        transformer = AggregatedTransformer()
        agg_df = transformer.transform(dataframe)

        # check either dataframe contain sales column
        with self.subTest():
            self.assertIn('sales', agg_df.columns)
        with self.subTest():
            self.assertEqual(agg_df.count(), dataframe.count())

    def test_mark_zero_neg_transformer(self):
        # create custom dataframe contain single column named sales
        columns = ["sales"]
        data = [20000, 100000, 3000, 0]
        spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
        rdd = spark.sparkContext.parallelize(data)
        df = rdd.map(lambda x: (x,)).toDF(columns)

        # calling the transformer
        transformer = MarkZeroNegTransformer()
        agg_df = transformer.transform(df)

        # converting the flag column into list
        cal_data = agg_df.select('flag').rdd.flatMap(lambda x: x).collect()

        # assert and check either dataframe contain 'flag' column or not
        with self.subTest():
            self.assertIn('flag', agg_df.columns)

        #  check either list contain 1 because dataframe contain 0 sale
        with self.subTest():
            self.assertIn(1, cal_data)

    def test_impute_mean_transformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()

        # read custom dataset
        dataframe = spark.read.csv('test_dataset/sales_data.csv', inferSchema=True, header=True)

        # calling the transformer
        transformer = ImputeMeanTransformer()
        agg_df = transformer.transform(dataframe)

        # counting the nulls in the returned dataset, there should be no null
        trans = agg_df.select(f.count(f.when(f.isnan('sales') | f.col('sales').isNull(), 'sales')).alias('sales'))
        count = trans.select('sales').rdd.flatMap(lambda x: x).collect()[0]
        self.assertEqual(count, 0)

    def test_train_test_transformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()

        # reading the dataframe
        dataframe = spark.read.csv('test_dataset/sales_data.csv', inferSchema=True, header=True)

        # calling the transformer
        transformer = TrainTestTransformer()
        train, test = transformer.transform(dataframe)

        # checking the size of the dataframe with original dataframe
        with self.subTest():
            self.assertEqual(math.ceil(dataframe.count() * 0.7) + 1, train.count())
        with self.subTest():
            self.assertEqual(math.floor(dataframe.count() * 0.3) - 1, test.count())


if __name__ == '__main__':
    unittest.main()
