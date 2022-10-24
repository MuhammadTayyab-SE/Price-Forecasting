from prophet import Prophet
import numpy as np
from hyperopt import fmin, tpe, Trials
import hyperopt


class Tune:

    def __init__(self, train, test, space):
        self.train = train
        self.test = test
        self.space = space

    def tune_hyper_parameters(self):
        space = self.space
        train_df = self.train.copy()
        test_df = self.test.copy()

        def objective(params):
            params = {
                'seasonality_mode': params['seasonality_mode'],
                'changepoint_prior_scale': float(params['changepoint_prior_scale']),
                'changepoint_range': float(params['changepoint_range']),
                'daily_seasonality': params['daily_seasonality'],
                'yearly_seasonality': params['yearly_seasonality'],
                'seasonality_prior_scale': float(params['seasonality_prior_scale'])
            }

            m = Prophet(**params)
            m.fit(train_df)
            daily_model_forecast_future_data = m.make_future_dataframe(periods=365, freq='D', include_history=False)
            forecast_df = m.predict(daily_model_forecast_future_data)

            y_true = test_df['y']
            y_predict = forecast_df['yhat']

            y_true = np.exp(y_true)
            y_predict = np.exp(y_predict)

            y_temp = y_true.copy()
            y_temp[y_temp == 0] = 1

            y_true.reset_index(inplace=True, drop=True)
            y_predict.reset_index(inplace=True, drop=True)
            y_temp.reset_index(inplace=True, drop=True)

            mape_value = np.mean(np.abs((y_true - y_predict) / y_temp)) * 100
            return mape_value

        trials = Trials()
        print(f"""Choosing best parameters for Store: 
            {self.test.store_id.unique()[0]} & Dept: {self.test.dept_id.unique()[0]}""")

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
        params = hyperopt.space_eval(space, best)
        print(f"""Best parameters for Store: {self.test.store_id.unique()[0]}
              & Dept: {self.test.dept_id.unique()[0]} are:\n {params}""")
        return params
