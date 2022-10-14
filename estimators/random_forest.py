from pyspark.ml import Estimator
from estimators.fb_prophet_estimator import mean_absolute_percentage_error
from pyspark.ml.param.shared import *
import warnings
import os
import joblib
import pandas as pd
from hyperopt import hp
from estimators.tune_random_forest import Tune
from sklearn.ensemble import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from transformers.string_indexer_transformer import StringIndexerTransformer

warnings.filterwarnings('ignore')


def get_features(df, input_cols: list(), output_col='features'):
    indexer = StringIndexerTransformer(input_cols=['store_id', 'dept_id'], output_cols=['store_id', 'dept_id'])
    df_transform = indexer.transform(df)
    df_transform = df_transform.fillna(0)
    vector_assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col, handleInvalid='skip')
    df_transform = vector_assembler.transform(df_transform)
    return df_transform


def get_string_indexer(df, input_cols: list(), output_cols:list(), original='discard'):
    string_indexer = StringIndexerTransformer(input_cols=input_cols, output_cols=output_cols, original=original)
    df = string_indexer.transform(df)
    return df


class RandomForestEstimator(Estimator, HasInputCol, HasOutputCols):
    def __int__(self):
        super(RandomForestEstimator, self).__int__()
        self.training_set = ""
        self.testing_set = ""

    def _fit(self, dataset):
        self.training_set = dataset.filter(dataset.split == "train")
        self.training_set = self.training_set.drop('split')

        self.testing_set = dataset.filter(dataset.split == 'test')
        self.testing_set = self.testing_set.drop('split')

        self.training_set = get_string_indexer(df=self.training_set, input_cols=['store_id', 'dept_id'],
                                               output_cols=['store_indexer', 'dept_indexer'])
        self.training_set = self.training_set.fillna(0)

        self.testing_set = get_string_indexer(df=self.testing_set, input_cols=['store_id', 'dept_id'],
                                              output_cols=['store_indexer', 'dept_indexer'])
        self.testing_set = self.testing_set.fillna(0)

        testing_pd = self.testing_set.toPandas()

        def forecast_sales(df):
            test = testing_pd[testing_pd.store_id == df.store_id.unique()[0]]
            test = test[test.dept_id == df.dept_id.unique()[0]]
            save_df = test.head(1)[['store_id', 'dept_id']]

            space = {
                'n_estimators': hp.quniform('n_estimators', 80, 160, 10),
                'max_depth': hp.quniform('max_depth', 10, 100, 10),
                'bootstrap': hp.choice('bootstrap', [True, False])
            }

            tune_hyp = Tune(train=df, test=test, space=space)
            param = tune_hyp.tune_hyper_parameters()
            param['n_estimators'] = int(param['n_estimators'])
            param['max_depth'] = int(param['max_depth'])
            train = df[['store_indexer', 'dept_indexer', 'wm_yr_wk', 'month', 'year', 'event_name_1',
                        'event_name_2', 'sales_lag_1', 'sales_lag_2', 'flag']]

            prediction_df = test[['store_id', 'dept_id', 'date', 'sales']]

            test = testing_pd[['store_indexer', 'dept_indexer', 'wm_yr_wk', 'month', 'year', 'event_name_1',
                               'event_name_2', 'sales_lag_1', 'sales_lag_2', 'flag']]

            rfr = RandomForestRegressor(**param)

            rfr.fit(train, df['sales'])
            predict = rfr.predict(test)
            prediction_df['prediction'] = pd.Series(predict)

            save_df['daily'] = mean_absolute_percentage_error(prediction_df['sales'], prediction_df['prediction'])

            df = prediction_df.copy()
            df.date = pd.to_datetime(df.date)
            df.set_index('date', inplace=True)
            df = df.resample('M').sum()
            df.reset_index(inplace=True)

            save_df['monthly'] = mean_absolute_percentage_error(df['sales'], df['prediction'])
            if os.path.exists("../models/decision_trees/MAPE.csv"):
                save_df.to_csv("../models/decision_trees/MAPE.csv", mode='a', index=False, header=False)
            else:
                save_df.to_csv("../models/decision_trees/MAPE.csv", mode='w', index=False)

            filename = "../models/decision_trees/" + prediction_df.head(1)['store_id'].tolist()[0] + '-' + \
                       prediction_df.head(1)['dept_id'].tolist()[0] + '.pkl'
            joblib.dump(rfr, filename)

            return prediction_df[['store_id', 'dept_id', 'date', 'sales', 'prediction']]

        data = self.training_set.groupby(['store_id', 'dept_id']).applyInPandas(forecast_sales,
                                                                                schema="""store_id string, dept_id string, date date, sales double, prediction double """)

        return data
