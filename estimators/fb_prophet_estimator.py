from prophet import Prophet
from pyspark.ml import Estimator
import pandas as pd
import numpy as np
import warnings
from hyperopt import hp
from estimators.tune_fb_prophet import Tune
import os
import joblib

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
        # changing columns according to prophet model requirements
        dataset = change_col_names(dataset, self.target_col)

        # extract train and test dataset from single dataframe
        df_train = dataset.filter(dataset.split == "train")
        df_test = dataset.filter(dataset.split == 'test').toPandas()

        def forecast_store_item(df):
            # create dataset according to training dataset
            test = df_test[df_test.store_id == df.store_id.unique()[0]]
            test = test[df_test.dept_id == df.dept_id.unique()[0]]
            test = test.iloc[:365, :]

            # predictions_df will be used to return predictions from the function
            predictions_df = test[['ds', 'store_id', 'dept_id', 'y']].copy()
            predictions_df.reset_index(inplace=True)

            # extracting columns needed for training of dataset
            df = df[['ds', 'y']]

            # save_df dataframe will be used to save scores into .csv file
            save_df = test.head(1)[['store_id', 'dept_id']]
            print(test)

            temp_df = df.copy()
            temp_test = test.copy()

            # defining space for hyperopt
            space = {
                'seasonality_mode': hp.choice('seasonality_mode', ['multiplicative', 'additive']),
                'changepoint_prior_scale': hp.choice('changepoint_prior_scale', np.arange(.1, 0.9, .1)),
                'changepoint_range': hp.choice('changepoint_range', np.arange(.1, 0.8, .1)),
                'daily_seasonality': hp.choice('daily_seasonality', [True, False]),
                'yearly_seasonality': hp.choice('yearly_seasonality', [True]),
                'weekly_seasonality': hp.choice('weekly_seasonality', [True]),
                'seasonality_prior_scale': hp.choice('seasonality_prior_scale', np.arange(1, 10, 1)),
            }
            #  tuning hyper-parameters
            tune = Tune(temp_df, temp_test, space)
            params = tune.tune_hyper_parameters()

            # train final model on tuned hps
            model = Prophet(**params)
            model.fit(df)

            # making future predictions
            daily_model_forecast_future_data = model.make_future_dataframe(periods=365, freq='D', include_history=False)
            daily_model_forecast = model.predict(daily_model_forecast_future_data)
            daily_model_forecast.reset_index(inplace=True)
            predictions_df['yhat'] = pd.Series(daily_model_forecast['yhat'].to_list())

            #  Calculating MAPE on daily bases
            df_ = predictions_df.copy()

            # taking anti-log of values for prediction
            df_['y'] = np.exp(df_.y)
            df_['yhat'] = np.exp(df_.yhat)

            daily_mape = mean_absolute_percentage_error(df_.y.copy(), df_.yhat.copy())
            print(f"Daily Model Store: {test.store_id.unique()[0]}, Dept:{test.dept_id.unique()[0]}")
            print(f"""Mean Absolute Percentage Error: 
                    {daily_mape} \n""")
            save_df['daily'] = daily_mape

            # calculating monthly MAPE
            df_.ds = pd.to_datetime(df_.ds)
            df_.set_index('ds', inplace=True)
            df_ = df_.resample('M').sum()
            monthly_mape = mean_absolute_percentage_error(df_['y'].copy(), df_['yhat'].copy())
            save_df['monthly'] = monthly_mape
            print(f"""Monthly MAPE: {monthly_mape} for store: {test.store_id.unique()[0]}
             & dept {test.dept_id.unique()[0]}""")

            # saving score in .csv file
            if os.path.exists("../models/prophet/MAPE.csv"):
                save_df.to_csv("../models/prophet/MAPE.csv", mode='a', index=False, header=False)
            else:
                save_df.to_csv("../models/prophet/MAPE.csv", mode='w', index=False)

            filename = "../models/prophet/" + predictions_df.head(1)['store_id'].tolist()[0] + '-' + \
                       predictions_df.head(1)['dept_id'].tolist()[0] + '.pkl'
            joblib.dump(model, filename)

            return predictions_df[['ds', 'store_id', 'dept_id', 'y', 'yhat']]

        results = df_train.groupby(['store_id', 'dept_id']).applyInPandas(forecast_store_item, schema=self.schema)
        return results
