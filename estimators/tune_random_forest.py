from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from estimators.fb_prophet_estimator import mean_absolute_percentage_error
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

        def objective(parameters):
            param = {
                'n_estimators': int(parameters['n_estimators']),
                'max_depth': int(parameters['max_depth']),
                'bootstrap': bool(parameters['bootstrap'])
            }

            train_x = train_df[['store_indexer', 'dept_indexer', 'wm_yr_wk', 'month', 'year', 'event_name_1',
                                'event_name_2', 'sales_lag_1', 'sales_lag_2', 'flag']]

            train_y = train_df[['sales']]

            test_x = test_df[['store_indexer', 'dept_indexer', 'wm_yr_wk', 'month', 'year', 'event_name_1',
                              'event_name_2', 'sales_lag_1', 'sales_lag_2', 'flag']]

            test_y = test_df['sales']

            rfr = RandomForestRegressor(**param)

            rfr.fit(train_x, train_y)
            predict = rfr.predict(test_x)
            predict = pd.Series(predict)
            mape_value = mean_absolute_percentage_error(test_y, predict)

            return mape_value

        trials = Trials()

        print(f"""Choosing best parameters for Store: 
            {self.test.store_id.unique()[0]} & Dept: {self.test.dept_id.unique()[0]}""")

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
        params = hyperopt.space_eval(space, best)
        print(f"""Best parameters for Store: {self.test.store_id.unique()[0]}
              & Dept: {self.test.dept_id.unique()[0]} are:\n {params}""")
        return params
