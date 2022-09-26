from pyspark.sql import SparkSession
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer

spark = SparkSession.builder.master("local[5]").appName('Feature Engineering').getOrCreate()
df = spark.read.csv("../Dataset/train-test/train.csv", header=True, inferSchema=True)
df.repartition(5)

log_transformer = LogTransformer()
log_df = log_transformer.transform(df)
log_df.show()

lag_transformer = LagTransformer(4)
transformed_df = lag_transformer.transform(log_df)
transformed_df.show()
