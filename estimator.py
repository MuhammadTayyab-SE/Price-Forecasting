from estimators.fb_prophet_estimator import ProphetEstimator
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f
import re

def get_original_columns(df_columns):
    """
    remove special characters form the string
    for example we have some string like [avg(count), avg(product)] this function will return [count, product]
    :param df_columns is the list of dataframe columns
    :return list of dataframe columns
    """
    new_list = list()
    for element in df_columns:
        try:
            new_list.append(re.sub(r'\W+', ' ', element).strip().split(' ')[1])
        except (Exception,):
            new_list.append(element)
    return new_list


spark = SparkSession.builder.appName('Estimator').getOrCreate()
df = spark.read.csv('file.csv', inferSchema=True, header=True)
split_col = f.split(df['timeStamp'], ' ')
df = df.withColumn('date', split_col.getItem(0)).drop('timeStamp')
df_agg = df.groupby('date').mean().orderBy('date')
columns = get_original_columns(df_agg.columns)
df_agg = df_agg.toDF(* columns)
prophet_model = ProphetEstimator('demand')
prophet_model = prophet_model.fit(df_agg)
