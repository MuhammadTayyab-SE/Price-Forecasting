from estimators.fb_prophet_estimator import ProphetEstimator
from estimators.random_forest import RandomForestEstimator
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f

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
df_cont = df_cont.orderBy(['store_id', 'dept_id', 'date'])

# prophet_model = ProphetEstimator()
# predicted_data = prophet_model.fit(df_cont)
# predicted_data.repartition(1).write.format('com.databricks.spark.csv').save('predicted', header=True)

tree_estimator = RandomForestEstimator()
tree_estimator.fit(df_cont)
