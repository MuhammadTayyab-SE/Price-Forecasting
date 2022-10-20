import unittest
from pyspark.sql import Row
from evaluator.mape_evaluator import MAPEEvaluator
from pyspark.sql.session import SparkSession
import math


class TestEvaluator(unittest.TestCase):

    def test_custom_evaluator(self):
        """
            I create a dataframe named df that contain two columns sales and prediction with custom values
            then call evaluate function of the MAPEEvalutor class that return mape value for that dataframe
        :return: None
        """
        spark = SparkSession.builder.appName('Unit Test').getOrCreate()

        # create spark dataframe
        df = spark.createDataFrame([
            Row(sales=10.0, prediction=5.0),
            Row(sales=5.0, prediction=4.0),
            Row(sales=4.0, prediction=3.0)
        ])

        #  calculating MAPE using evaluate function.
        mape = MAPEEvaluator(prediction_col='prediction', label_col='sales', logged_values=False).evaluate(df)

        self.assertEqual(math.floor(mape), 31)


if __name__ == '__main__':
    unittest.main()
