from prophet import Prophet
from pyspark.ml import Estimator
from pyspark.sql.pandas.functions import PandasUDFType, pandas_udf


def change_col_names(dataframe, target_column: str):
    """This function rename column names according to FB prophet model convention.
    param : dataframe and target column
    :return: dataframe
    """

    dataframe = dataframe.withColumnRenamed(target_column, 'y')
    dataframe = dataframe.withColumnRenamed('date', 'ds')
    return dataframe


class ProphetEstimator(Estimator):
    """
    This class is custom estimator of prophet model and also inherits the estimator class
    """

    def __init__(self, target_col: str):
        super().__init__()
        self.target_col = target_col
        self.prophet = Prophet(interval_width=0.95, yearly_seasonality=True)

    def _fit(self, dataset):
        df = change_col_names(dataset, self.target_col)
        df = df.select(['ds', 'y']).toPandas()
        df.head(20)
        return self.prophet.fit(df)

    def getParam(self):
        return self.prophet.params

