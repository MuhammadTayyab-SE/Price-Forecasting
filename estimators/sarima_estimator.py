from pyspark.ml import Estimator
from pyspark.ml.param.shared import *
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pmdarima as pm
import numpy as np
import pandas as pd
import os
import joblib


def mean_absolute_percentage_error(y_true, y_predict):
    """
    This function calculate the MAPE value for two pandas series
    :param y_true: actual values in pandas series format
    :param y_predict: predicted values in pandas series format
    :return float: return calculated MAPE value
    """
    print(y_true)
    print(y_predict)
    y_true.reset_index(inplace=True, drop=True)
    y_predict.reset_index(inplace=True, drop=True)
    y_temp = y_true.copy()
    y_temp[y_temp == 0] = 1
    print(y_temp[y_temp == 0])

    return np.mean(np.abs((y_true - y_predict) / y_temp)) * 100


def get_number_of_lags(array):
    i = 1
    for value in array:
        if value < 0.05:
            return i
        i += 1


def is_time_series_stationary(df, col_name):
    """
        This function check either time series is stationary or not and return accordingly
    :param df: pyspark dataframe
    :param col_name: column for which we want to file p-value
    :return bool: value indicates either series is stationary or not
    :return float: calculated p-value using dickey fuller test
    """
    df_ = df[col_name].copy()
    p_value = adfuller(df_)[1]
    if p_value <= 0.05:
        return True, p_value
    else:
        return False, p_value


def find_d(df, col_name):
    """

    :param df: pyspark dataframe
    :param col_name:
    :return:
    """
    df_ = df[col_name].copy()
    difference = 1
    while True:
        p_value = adfuller(df_.diff(difference).dropna())[1]
        if p_value <= 0.05:
            return difference
        difference += 1


class SarimaEstimator(Estimator, HasInputCol, HasOutputCols):

    def __init__(self):
        super().__init__()
        self.schema = """date date, store_id string, dept_id string, sales double, prediction double"""

    def _fit(self, dataset):
        train_data = dataset.filter(dataset.split == "train")
        test_data = dataset.filter(dataset.split == 'test')
        test_pd = test_data.toPandas()

        def forecast_sales(df):
            """
            This function train, hyperparameter tune and saves model
            :param df: pyspark dataframe that have data chunks according to group by
            :return df: desired dataframe
            """
            # prepare test dataset according to the training dataset
            df_test = test_pd[test_pd.store_id == df.store_id.unique()[0]]
            df_test = df_test[df_test.dept_id == df.dept_id.unique()[0]]

            # save_df dataframe will be used to store score in csv file
            save_df = df_test.head(1)[['store_id', 'dept_id']]
            save_df.reset_index(inplace=True, drop=True)

            # prediction df returned by forecast_sales function
            predictions_df = df_test[['date', 'store_id', 'dept_id', 'sales']].copy()
            predictions_df.reset_index(inplace=True, drop=True)
            predictions_df = predictions_df.loc[:364, :]

            # check either time series is stationary or not
            condition, p_value = is_time_series_stationary(df, 'sales')

            d = None
            #  if time series is not stationary then find the value of d

            # get number of lags
            pacf, ci = sm.tsa.pacf(df.sales, alpha=0.05)
            lags = get_number_of_lags(pacf)

            if not condition:
                # finding the value to differencing to make time series stationary
                d = find_d(df, 'sales')

            # auto-arima model setup
            s_model = pm.auto_arima(df.sales, p=lags, q=lags,
                                    test='adf',
                                    m=12,
                                    start_P=0,
                                    start_Q=0,
                                    D=1, d=d, seasonal=True,
                                    trace=True,
                                    error_action='ignore',
                                    alpha=0.05,
                                    suppress_warnings=True,
                                    stepwise=True)

            # predicting the 365 days.
            n_periods = 365
            fitted, conf = s_model.predict(n_periods=n_periods, return_conf_int=True)

            # converting pandas series to pandas dataframe.
            prediction = pd.DataFrame(fitted, columns=['prediction_mean'])
            prediction.reset_index(inplace=True, drop=True)
            predictions_df['prediction'] = prediction['prediction_mean']

            df_ = df_test[['date', 'sales']].copy()
            df_.reset_index(inplace=True, drop=True)
            df_['prediction'] = prediction['prediction_mean']
            df_ = df_.loc[:364, :]

            # taking anti-log of values for prediction
            df_['sales'] = np.exp(df_.sales)
            df_['prediction'] = np.exp(df_.prediction)

            # calculating MAPE value of daily modeling
            daily_mape = mean_absolute_percentage_error(df_['sales'], df_['prediction'])
            save_df['daily'] = daily_mape
            print(f"Daily MAPE: {daily_mape} for store: {df.store_id.unique()[0]} & dept {df.dept_id.unique()[0]}")

            # calculating monthly MAPE value
            df_.date = pd.to_datetime(df_.date)
            df_.set_index('date', inplace=True)
            df_ = df_.resample('M').sum()
            monthly_mape = mean_absolute_percentage_error(df_['sales'].copy(), df_['prediction'].copy())
            save_df['monthly'] = monthly_mape
            print(f"Monthly MAPE: {monthly_mape} for store: {df.store_id.unique()[0]} & dept {df.dept_id.unique()[0]}")

            # saving MAPEs to .csv file
            if os.path.exists("../models/sarima/MAPE.csv"):
                save_df.to_csv("../models/sarima/MAPE.csv", mode='a', index=False, header=False)
            else:
                save_df.to_csv("../models/sarima/MAPE.csv", mode='w', index=False)

            # saving sarima model to .pkl file
            filename = "../models/sarima/" + df.head(1)['store_id'].tolist()[0] + '-' + \
                       df.head(1)['dept_id'].tolist()[0] + '.pkl'
            joblib.dump(s_model, filename)

            del s_model, df_,
            # return desired columns from dataframe
            return predictions_df[['date', 'store_id', 'dept_id', 'sales', 'prediction']]

        predictions = train_data.groupby(['store_id', 'dept_id'])\
                                .applyInPandas(forecast_sales, schema=self.schema)
        return predictions
