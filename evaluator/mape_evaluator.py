from pyspark.ml.evaluation import Evaluator
import pyspark.sql.functions as f
import numpy as np


def mean_absolute_percentage_error(y_true, y_predict):
    y_temp = y_true.copy()
    y_temp[y_temp == 0] = 1
    y_true.reset_index(inplace=True, drop=True)
    y_predict.reset_index(inplace=True, drop=True)
    y_temp.reset_index(inplace=True, drop=True)
    return np.mean(np.abs((y_true - y_predict) / y_temp)) * 100


class MAPEEvaluator(Evaluator):

    def __init__(self, prediction_col, label_col):
        super().__init__()
        self.predictionCol = prediction_col
        self.labelCol = label_col

    def evaluate(self, dataset):
        # calculating the anti-log for label/sales column
        dataset = dataset.withColumn(self.labelCol, f.lit(10 ** dataset[self.labelCol]))

        # calculating the anti-log for prediction column
        dataset = dataset.withColumn(self.predictionCol, f.lit(10 ** dataset[self.predictionCol]))

        y_true = dataset.select(self.labelCol).toPandas()
        y_predict = dataset.select(self.predictionCol).toPandas()
        return mean_absolute_percentage_error(y_true=y_true, y_predict=y_predict)
