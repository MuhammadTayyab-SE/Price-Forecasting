from pyspark.sql import SparkSession
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer


spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


# Loading Dataset
df = spark.read.parquet("../df_final.parquet.gzip")
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


# log transformation on train dataset

log_transformer = LogTransformer()
log_df_train = log_transformer.transform(train_set)
log_df_train.show()

# lag transformation on train dataset

lag_transformer = LagTransformer(offset=2)
transformed_df_train = lag_transformer.transform(log_df_train)
transformed_df_train.show()

# log transformation to test dataset

log_transformer = LogTransformer()
log_df_test = log_transformer.transform(test_set)
log_df_test.show()

# lag transformation to test dataset

lag_transformer = LagTransformer(offset=2)
transformed_df_test = lag_transformer.transform(log_df_test)
transformed_df_test.show()
