from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import *
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.aggregated_transformer import save_aggregated_data
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer


spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


# Loading Dataset

df = spark.read.parquet("../df.parquet.gzip")
df.repartition(5)
df = df.drop(*['snap_CA', 'snap_TX', 'snap_WI'])

aggregated_transformer = AggregatedTransformer()
aggregated_df = aggregated_transformer.transform(df)
aggregated_df.show(3)


zero_ng_transformer = MarkZeroNegTransformer()
zero_ng_df = zero_ng_transformer.transform(aggregated_df)
zero_ng_df.show(1)


train_test_transformer = TrainTestTransformer()
train_set, test_set = train_test_transformer.transform(zero_ng_df)

train_set.repartition(1).write.format('com.databricks.spark.csv').save('train.csv', header=True)
test_set.repartition(1).write.format('com.databricks.spark.csv').save('test.csv', header=True)

# mean_transformer = ImputeMeanTransformer()
# mean_df = mean_transformer.transform(zero_ng_df)
# mean_df.show(1)




