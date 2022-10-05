from prophet import Prophet
from pyspark.ml import Estimator
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def change_col_names(dataset, target_col):
    dataset = dataset.withColumnRenamed('date', 'ds')
    dataset = dataset.withColumnRenamed(target_col, 'y')
    return dataset


class ProphetEstimator(Estimator):
    """
    This class is custom estimator of prophet model and also inherits the estimator class
    """

    def __init__(self):
        super().__init__()
        self.target_col = 'demand'
        self.schema  = """ds date, store_id string, dept_id string, y double, yhat double"""

    def _fit(self, dataset):
        dataset = change_col_names(dataset, self.target_col)

        df_train = dataset.filter(dataset.split == "train")
        df_test = dataset.filter(dataset.split == 'test').toPandas()

        # @pandas_udf(returnType=DataFrame)
        def forecast_store_item(df):
            test = df_test[df_test.store_id == df.store_id.unique()[0]]
            test = test[df_test.dept_id == df.dept_id.unique()[0]]
            test = test.iloc[:365, :]
            df = df[['ds', 'y']]

            #     instantiate the model, configure the parameters
            model = Prophet(
                interval_width=0.95,
                changepoint_prior_scale=0.1,
                changepoint_range=0.5,
                seasonality_mode='additive'
            )
            model.fit(df)

            daily_model_forecast_future_data = model.make_future_dataframe(periods=365, freq='D', include_history=False)
            daily_model_forecast = model.predict(daily_model_forecast_future_data)

            print(f"Daily Model Store: {test.store_id.unique()[0]}, Dept:{test.dept_id.unique()[0]}")
            print(f"Mean Squared Error: {mean_squared_error(test.y, daily_model_forecast.yhat)}")
            print(f"R2 Score: {r2_score(test.y, daily_model_forecast.yhat)}")
            print(f"Mean Absolute Error: {mean_absolute_error(test.y, daily_model_forecast.yhat)}")
            print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(test.y, daily_model_forecast.yhat)} \n")

            test_parms = test[['ds', 'store_id', 'dept_id', 'y']]
            test_parms.reset_index(inplace=True)
            daily_model_forecast.reset_index(inplace=True)
            test_parms['yhat'] = pd.Series(daily_model_forecast['yhat'].to_list())

            # fig = px.line(test_parms, x="ds", y=['y', 'yhat'],
            #               title=f"Daily Model Store: {test_parms.store_id.unique()[0]}, Dept:{test_parms.dept_id.unique()[0]} ")
            # fig.show()

            return test_parms[['ds', 'store_id', 'dept_id', 'y', 'yhat']]

        results = df_train.groupby('store_id', 'dept_id').applyInPandas(forecast_store_item, schema=self.schema)
        return results

    def get_param(self):
        return self.prophet.params
