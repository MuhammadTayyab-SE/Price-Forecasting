from pyspark.sql import SparkSession
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import *
from transformers.aggregated_transformer import AggregatedTransformer
from transformers.impute_mean_transformer import ImputeMeanTransformer
from transformers.train_test_transformer import TrainTestTransformer
from transformers.mark_zero_neg_transformer import MarkZeroNegTransformer


spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


def get_file_path(url):
    return 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

# Loading Dataset


# calender = 'https://drive.google.com/file/d/1wHylk7Im9FBz9Pfm5Lvh8mScBe_9zKE2/view?usp=sharing'
# sales_eval = 'https://drive.google.com/file/d/1Cku6DtQXf9sgXkg0GExBqcLepEvH9nqL/view?usp=sharing'
# sales_val = 'https://drive.google.com/file/d/1PWycNlOKbyqPRBYAzEprhbyp_ED5xeLv/view?usp=sharing'
# prices = 'https://drive.google.com/file/d/17Y45Kkela5Uag1wYPz7ePRneUv5Jus9M/view?usp=sharing'

calendar_df = spark.read.csv('../Dataset/calendar.csv', inferSchema=True, header=True)
sales_df_eval = spark.read.csv('../Dataset/sales_train_evaluation.csv', inferSchema=True, header=True)
sales_df_val = spark.read.csv('../Dataset/sales_train_validation.csv', inferSchema=True, header=True)
prices_df = spark.read.csv('../Dataset/sell_prices.csv', inferSchema=True, header=True)

# Printing Dataset information
print(f"Total stores: {prices_df.select('store_id').distinct().count()}")
prices_df.select('store_id').distinct().orderBy('store_id').show()

print(f"Total departments: {sales_df_eval.select('dept_id').distinct().count()}")
sales_df_eval.select('dept_id').distinct().orderBy('dept_id').show()

print(f"Total cat: {sales_df_eval.select('cat_id').distinct().count()}")
sales_df_eval.select('cat_id').distinct().orderBy('cat_id').show()

print(f"Total items: {prices_df.select('item_id').distinct().count()}")

agg_transformer = AggregatedTransformer()
aggregated_df = agg_transformer.transform(sales_df_val)
aggregated_df.show(1)

zero_ng_transformer = MarkZeroNegTransformer()
zero_ng_df = zero_ng_transformer.transform(aggregated_df)
zero_ng_df.show(1)

mean_transformer = ImputeMeanTransformer()
mean_df = mean_transformer.transform(zero_ng_df)
mean_df.show(1)


train_test_transformer = TrainTestTransformer()
train, test = train_test_transformer.transform(zero_ng_df)
train.show(1)
test.show(1)
