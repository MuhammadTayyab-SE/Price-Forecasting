from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from feature_transformers.lag_feature import LagTransformer

spark = SparkSession.builder.master("local[5]").appName('Feature Engineering').getOrCreate()
df = spark.read.csv("../Dataset/train.csv", header=True, inferSchema=True)
df.repartition(5)
partition = Window.partitionBy("sales").orderBy("date")

temp = df.withColumn('Lag_feature', F.row_number().over(partition))
temp.show()