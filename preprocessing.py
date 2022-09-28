from pyspark.sql import SparkSession
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer


spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


# Loading Dataset
df = spark.read.parquet("../df.parquet.gzip")
df.repartition(5)

# drop unnecessary columns
df = df.drop(*['snap_CA', 'snap_TX', 'snap_WI'])

# aggregate data on store department level
aggregated_transformer = AggregatedTransformer()
aggregated_df = aggregated_transformer.transform(df)
aggregated_df.show(3)

# mark rows with zero sales
zero_ng_transformer = MarkZeroNegTransformer()
zero_ng_df = zero_ng_transformer.transform(aggregated_df)
zero_ng_df.show(1)

# split data into train and test
train_test_transformer = TrainTestTransformer()
train_set, test_set = train_test_transformer.transform(zero_ng_df)

# fill nans with mean value of that store
mean_transformer = ImputeMeanTransformer()
train_set = mean_transformer.transform(train_set)
test_set = mean_transformer.transform(test_set)

# saving train test data in csv file
train_set.repartition(1).write.format('com.databricks.spark.csv').save('train', header=True)
test_set.repartition(1).write.format('com.databricks.spark.csv').save('test', header=True)
