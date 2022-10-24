import pyspark.sql.functions as f
import pandas as pd
import numpy as np


class SelectModel:
    def __init__(self, spark):
        #  loading all files
        self.prophet = spark.read.csv()
        self.forest = spark.read()
        self.sarima = spark.read.csv()
