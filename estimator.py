from estimators.fb_prophet_estimator import ProphetEstimator
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.param import Params

import warnings
warnings.filterwarnings('ignore')

spark = SparkSession.builder.appName('Estimator').getOrCreate()
df_train = spark.read.csv('../Dataset/train.csv', inferSchema=True, header=True)
df_train = df_train.withColumn('date', f.to_date(df_train.date))
df_train = df_train.withColumn('split', f.lit('train'))

df_test = spark.read.csv('../Dataset/test.csv', inferSchema=True, header=True)
df_test = df_test.withColumn('date', f.to_date(df_test.date))
df_test = df_test.withColumn('split', f.lit('test'))

df_cont = df_train.union(df_test)

prophet_model = ProphetEstimator()
prophet_model = prophet_model.fit(df_cont)
prophet_model.count()
