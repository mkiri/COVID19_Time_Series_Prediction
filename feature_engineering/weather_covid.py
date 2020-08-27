# Databricks notebook source
# This segment of code is meant to merge all features from various datasets into an unified table. This table will be fed through various models.
# Databricks notebook source
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline


# COMMAND ----------

df = spark.table("city_temperature_1_csv")

# COMMAND ----------

df.show(20)

# COMMAND ----------

lister = 'Italy','Canada'

# COMMAND ----------

from pyspark.sql.functions import concat, col, lit


df_g7 = df \
  .filter((df.Country=='Canada') |(df.Country=='Italy') | (df.Country=='Japan') | (df.Country=='United Kingdom') | (df.Country=='Germany') | (df.Country=='France') & (df.Year=='2019')) \
  .select(concat(col("Month"), lit(" "), col("Day")).alias("Date"),"Country","Month","Day","AvgTemperature",'City')\

# COMMAND ----------

df_g7 = df_g7.filter(df_g7.AvgTemperature>-20)

# COMMAND ----------

df_g7.show(20)

# COMMAND ----------

display(df_g7)

# COMMAND ----------

df_Canada = df_g7 \
  .filter((df_g7.Country=='Canada')) \
  .select("Date","AvgTemperature")\

# COMMAND ----------

#importing window lib
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql.functions import avg
from pyspark.sql.functions import desc
from pyspark.sql import Row


df_Canada = df_Canada.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_Canada=df_Canada.withColumn("temp", avg('AvgTemperature').over(window)) \

df_Canada = df_Canada.dropDuplicates((['Date']))\
.orderBy('id') \

# COMMAND ----------

# Looking to see if all year round data is present
#df_Canada.show(1000)

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------



# COMMAND ----------

df_Canada.show()

# COMMAND ----------

df_Italy = df_g7 \
  .filter((df_g7.Country=='Italy')) \
  .select("Date","AvgTemperature")\


# COMMAND ----------

df_Germany = df_g7 \
  .filter((df_g7.Country=='Germany')) \
  .select("Date","AvgTemperature")\


# COMMAND ----------

df_France = df_g7 \
  .filter((df_g7.Country=='France')) \
  .select("Date","AvgTemperature")\


# COMMAND ----------

df_Japan = df_g7 \
  .filter((df_g7.Country=='Japan')) \
  .select("Date","AvgTemperature")\


# COMMAND ----------

df_UK = df_g7 \
  .filter((df_g7.Country=='United Kingdom'))\
  .select("Date","AvgTemperature")\


# COMMAND ----------

df_Germany.show(1000)

# COMMAND ----------

df_UK = df_UK.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_UK=df_UK.withColumn("temp", avg('AvgTemperature').over(window)) \

df_UK = df_UK.dropDuplicates((['Date']))\
.orderBy('id') \

df_Germany = df_Germany.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_Germany=df_Germany.withColumn("temp", avg('AvgTemperature').over(window)) \

df_Germany = df_Germany.dropDuplicates((['Date']))\
.orderBy('id') \

df_Italy = df_Italy.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_Italy=df_Italy.withColumn("temp", avg('AvgTemperature').over(window)) \

df_Italy = df_Italy.dropDuplicates((['Date']))\
.orderBy('id') \

df_France = df_France.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_France=df_France.withColumn("temp", avg('AvgTemperature').over(window)) \

df_France = df_France.dropDuplicates((['Date']))\
.orderBy('id') \

df_Japan = df_Japan.select("*").withColumn("id", monotonically_increasing_id())
#creating window that partitions by season
window = Window \
.partitionBy("Date") \

df_Japan=df_Japan.withColumn("temp", avg('AvgTemperature').over(window)) \

df_Japan = df_Japan.dropDuplicates((['Date']))\
.orderBy('id') \


# COMMAND ----------

df_Germany.show()

# COMMAND ----------

df_UK = df_UK.selectExpr("Date as Date", "temp as temp_uk")

df_Japan = df_Japan.selectExpr("Date as Date", "temp as temp_japan")

df_France = df_France.selectExpr("Date as Date", "temp as temp_france")

df_Germany = df_Germany.selectExpr("Date as Date", "temp as temp_germany")

df_Italy = df_Italy.selectExpr("Date as Date", "temp as temp_italy")

df_Canada = df_Canada.selectExpr("Date as Date", "temp as temp_canada")

# COMMAND ----------

UK = df_UK.toPandas()

# COMMAND ----------

Canada = df_Canada.toPandas()

# COMMAND ----------

Germany = df_Germany.toPandas()

# COMMAND ----------

Japan =df_Japan.toPandas()

# COMMAND ----------

France = df_France.toPandas()

# COMMAND ----------

Italy = df_Italy.toPandas()

# COMMAND ----------

dfs = [Canada, UK, Italy, Japan, Germany, France]

# COMMAND ----------

from functools import reduce
df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)

# COMMAND ----------

display(df_final)

# COMMAND ----------


