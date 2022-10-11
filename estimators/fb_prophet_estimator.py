from prophet import Prophet
from pyspark.ml import Estimator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
from hyperopt import hp
from estimators.tune_fb_prophet import Tune

warnings.filterwarnings('ignore')


def change_col_names(dataset, target_col):
    dataset = dataset.withColumnRenamed('date', 'ds')
    dataset = dataset.withColumnRenamed(target_col, 'y')
    return dataset


def mean_absolute_percentage_error(y_true, y_predict):
    y_temp = y_true.copy()
    y_temp[y_temp == 0] = 1
    y_true.reset_index(inplace=True, drop=True)
    y_predict.reset_index(inplace=True, drop=True)
    y_temp.reset_index(inplace=True, drop=True)
    return np.mean(np.abs((y_true - y_predict) / y_temp)) * 100


class ProphetEstimator(Estimator):
    """
    This class is custom estimator of prophet model and also inherits the estimator class
    """

    def __init__(self):
        super().__init__()
        self.target_col = 'demand'
        self.schema = """ds date, store_id string, dept_id string, y double, yhat double"""
        # self.schema = """ds date,y double"""

    def _fit(self, dataset):
        dataset = change_col_names(dataset, self.target_col)
        df_train = dataset.filter(dataset.split == "train")
        df_test = dataset.filter(dataset.split == 'test').toPandas()

        def forecast_store_item(df):
            test = df_test[df_test.store_id == df.store_id.unique()[0]]
            test = test[df_test.dept_id == df.dept_id.unique()[0]]
            test = test.iloc[:365, :]
            df = df[['ds', 'y']]

            temp_df = df.copy()
            temp_test = test.copy()

            space = {
                'seasonality_mode': hp.choice('seasonality_mode', ['multiplicative', 'additive']),
                'changepoint_prior_scale': hp.choice('changepoint_prior_scale', np.arange(.1, 0.9, .1)),
                'changepoint_range': hp.choice('changepoint_range', np.arange(.1, 0.8, .1)),
                'daily_seasonality': hp.choice('daily_seasonality', [True]),
                'yearly_seasonality': hp.choice('yearly_seasonality', [True]),
                'weekly_seasonality': hp.choice('weekly_seasonality', [True]),
                'seasonality_prior_scale': hp.choice('seasonality_prior_scale', np.arange(1, 10, 1)),
            }

            tune = Tune(temp_df, temp_test, space)
            params = tune.tune_hyper_parameters()

            model = Prophet(**params)
            model.fit(df)

            daily_model_forecast_future_data = model.make_future_dataframe(periods=365, freq='D', include_history=False)
            daily_model_forecast = model.predict(daily_model_forecast_future_data)

            print(f"Daily Model Store: {test.store_id.unique()[0]}, Dept:{test.dept_id.unique()[0]}")
            print(f"Mean Squared Error: {mean_squared_error(test.y, daily_model_forecast.yhat)}")
            print(f"R2 Score: {r2_score(test.y, daily_model_forecast.yhat)}")
            print(f"Mean Absolute Error: {mean_absolute_error(test.y, daily_model_forecast.yhat)}")
            print(f"""Mean Absolute Percentage Error: 
                    {mean_absolute_percentage_error(test.y, daily_model_forecast.yhat)} \n""")
            test_params = test[['ds', 'store_id', 'dept_id', 'y']]
            test_params.reset_index(inplace=True)
            daily_model_forecast.reset_index(inplace=True)
            test_params['yhat'] = pd.Series(daily_model_forecast['yhat'].to_list())

            return test_params[['ds', 'store_id', 'dept_id', 'y', 'yhat']]

        results = df_train.groupby(['store_id', 'dept_id']).applyInPandas(forecast_store_item, schema=self.schema)
        return results
