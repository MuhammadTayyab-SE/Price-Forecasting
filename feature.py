from pyspark.sql import SparkSession
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer

spark = SparkSession.builder.master("local[5]").appName('Feature Engineering').getOrCreate()

# feature transformation on train data
df_train = spark.read.csv("../Dataset/train.csv", header=True, inferSchema=True)
df_train.repartition(5)

log_transformer = LogTransformer(column=['sales'])
log_df_train = log_transformer.transform(df_train)
log_df_train.show()

lag_transformer = LagTransformer(4)
transformed_df_train = lag_transformer.transform(log_df_train)
transformed_df_train.show()

# feature transformation on test data

df_test = spark.read.csv("../Dataset/test.csv", header=True, inferSchema=True)
df_test.repartition(5)

log_transformer = LogTransformer(column=['sales'])
log_df_test = log_transformer.transform(df_test)
log_df_test.show()

lag_transformer = LagTransformer(4)
transformed_df_test = lag_transformer.transform(log_df_test)
transformed_df_test.show()
