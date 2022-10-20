from pyspark.ml.evaluation import Evaluator
import pyspark.sql.functions as f
import numpy as np


def mean_absolute_percentage_error(y_true, y_predict):
    """
    This function calculate MAPE value for given pandas series
    :param y_true: actual values
    :param y_predict: predicted values
    :return: float mape value
    """
    y_temp = y_true.copy()
    y_temp[y_temp == 0] = 1

    #  resting series index
    y_true.reset_index(inplace=True, drop=True)
    y_predict.reset_index(inplace=True, drop=True)
    y_temp.reset_index(inplace=True, drop=True)

    # calculating MAPE values
    mape_value = np.mean(np.abs((y_true - y_predict) / y_temp)) * 100
    return mape_value


class MAPEEvaluator(Evaluator):

    def __init__(self, prediction_col, label_col, logged_values=True):
        super().__init__()
        self.predictionCol = prediction_col
        self.labelCol = label_col
        self.logged_values = logged_values

    def _evaluate(self, dataset):
        # check either dataframe contain logged values or not
        if self.logged_values:
            # calculating the anti-log for label/sales column
            dataset = dataset.withColumn(self.labelCol, f.lit(10 ** dataset[self.labelCol]))
            # calculating the anti-log for prediction column
            dataset = dataset.withColumn(self.predictionCol, f.lit(10 ** dataset[self.predictionCol]))

        # converting pyspark dataframe into pandas series
        y_true = dataset.select(self.labelCol).toPandas()[self.labelCol]
        y_predict = dataset.select(self.predictionCol).toPandas()[self.predictionCol]

        # calling mape function that calculate MAPE for two series
        mape = mean_absolute_percentage_error(y_true=y_true, y_predict=y_predict)
        return mape
