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

# COMMAND ----------

# DBTITLE 1,Load Data (train.csv)
# LOAD DATA (train.csv)

# Note: CSVs have already been loaded using the Databricks UI
train = spark.table("train_csv")
print(train.dtypes)
train.show(10)

# COMMAND ----------

# DBTITLE 1,Only consider confirmed cases
# ONLY CONSIDER CONFIRMED CASES

train_cc = train \
  .filter(train.Target=="ConfirmedCases") \
  .select("Date", "Country_Region", "Province_State", "TargetValue")\

# COMMAND ----------

# DBTITLE 1,Check for invalid values
# CHECK FOR INVALID VALUES (nulls, empty strings, negative numbers)

num_invalid_date = train_cc.filter(train_cc.Date.isNull()).count()
num_invalid_country = train_cc.filter(train_cc.Country_Region.isNull() | (train_cc.Country_Region=="")).count()
num_invalid_state = train_cc.filter(train_cc.Province_State.isNull() | (train_cc.Province_State=="")).count()
num_invalid_target = train_cc.filter(train_cc.TargetValue.isNull() | (train_cc.TargetValue=="") | (train_cc.TargetValue<0)).count()

print("Number of invalid Date values: %d" % num_invalid_date)
print("Number of invalid Country_Region values: %d" % num_invalid_country)
print("Number of invalid Province_State values: %d" % num_invalid_state)
print("Number of invalid TargetValue values: %d" % num_invalid_target)


# COMMAND ----------

# Why are there negative confirmed cases? --> Look into this more later
train_cc \
  .filter(train_cc.TargetValue.isNull() | (train_cc.TargetValue=="") | (train_cc.TargetValue<0)) \
  .groupBy("Country_Region")\
  .agg(sum("TargetValue").alias("ttl_cc"), min("Date"), max("Date"), min("TargetValue"), max("TargetValue"))\
  .orderBy("Country_Region")\
  .show(10)

# COMMAND ----------

train_cc.show()

# COMMAND ----------

# DBTITLE 1,Handing missing state values
# HANDLING MISSING STATE VALUES

# Why are there missing province/state values?
# Province/State should only be populated for some countries

# Check US
num_invalid_state_us = train_cc.filter((train_cc.Country_Region=="US") & (train_cc.Province_State.isNull() | (train_cc.Province_State==""))).count()
print("Number of invalid Province_State values for US: %d" % num_invalid_state_us)

# Check Canada
num_invalid_state_ca = train_cc.filter((train_cc.Country_Region=="Canada") & (train_cc.Province_State.isNull() | (train_cc.Province_State==""))).count()
print("Number of invalid Province_State values for CA: %d" % num_invalid_state_ca)

# COMMAND ----------

# Why are there missing province/state values?
train_cc \
  .filter((train_cc.Country_Region=="Canada") & (train_cc.Date>'2020-06-01'))\
  .orderBy("Date")\
  .show()

# Answer: When province is null, it shows the total value for the country

# COMMAND ----------

# DBTITLE 1,Country-level data set
# GENERATE COUNTRY-LEVEL DATA SET

# Confirmed cases by country
train_cc_country = train \
  .filter((train.Target=="ConfirmedCases") & train.Province_State.isNull()) \
  .select("Date", "Country_Region", "TargetValue")\

# Top 5 countries in terms of confirmed cases
train_cc_country \
  .groupBy("Country_Region")\
  .agg(sum("TargetValue").alias("ttl_cc"), min("Date"), max("Date"))\
  .orderBy("ttl_cc", ascending=False)\
  .show(5)

# COMMAND ----------

# DBTITLE 1,State-level data set (US only)
# GENERATE STATE-LEVEL DATA SET (US ONLY)

# Confirmed cases by state (US Only)
train_cc_us_state = train \
  .filter((train.Target=="ConfirmedCases") & (train.Country_Region=="US") & train.Province_State.isNotNull() & train.County.isNull()) \
  .select("Date", "Province_State", "TargetValue")\

# Top 5 states in terms of confirmed cases
train_cc_us_state \
  .groupBy("Province_State")\
  .agg(sum("TargetValue").alias("ttl_cc"), min("Date"), max("Date"))\
  .orderBy("ttl_cc", ascending=False)\
  .show(5)

# COMMAND ----------

# DBTITLE 1,Check for missing dates
# CHECK FOR MISSING DATES (country-level)

df_lag = train_cc_country \
  .withColumn('prev_date', lag(col("Date")).over(Window.partitionBy("Country_Region").orderBy("Date"))) \
  .withColumn("date_diff", when(col("prev_date").isNotNull(), datediff(col("Date"),col("prev_date"))).otherwise(0))\

# df_lag.orderBy("Country_Region", "Date").show()

df_lag \
  .filter(col("prev_date").isNotNull() & (col("date_diff")>1)) \
  .count()

# Count is 0, therefore no missing dates

# COMMAND ----------

# CHECK FOR MISSING DATES (US state-level)

df_lag = train_cc_us_state \
  .withColumn('prev_date', lag(col("Date")).over(Window.partitionBy("Province_State").orderBy("Date"))) \
  .withColumn("date_diff", when(col("prev_date").isNotNull(), datediff(col("Date"),col("prev_date"))).otherwise(0))\

# df_lag.orderBy("Province_State", "Date").show()

df_lag \
  .filter(col("prev_date").isNotNull() & (col("date_diff")>1)) \
  .count()

# Count is 0, therefore no missing dates

# COMMAND ----------

# DBTITLE 1,Handle negative target values
# HANDLE NEGATIVE TARGET VALUES (Continued from above)

train_cc \
  .filter(train_cc.TargetValue.isNull() | (train_cc.TargetValue=="") | (train_cc.TargetValue<0)) \
  .show(10)

# COMMAND ----------

# We are interested in US and Canada values
train_cc_country \
  .filter((train_cc_country.TargetValue.isNull() | (train_cc_country.TargetValue=="") | (train_cc_country.TargetValue<0)) \
         & train_cc_country.Country_Region.isin("Canada","US")) \
  .count()
# No invalid values for Canada and US (country-level)

# COMMAND ----------

# What countries have invalid TargetValues?
train_cc_country \
  .filter(train_cc_country.TargetValue.isNull() | (train_cc_country.TargetValue=="") | (train_cc_country.TargetValue<0)) \
  .groupBy("Country_Region")\
  .count()\
  .orderBy("Country_Region")\
  .show()
# Most countries with invalid values only have one or two days of errors

# COMMAND ----------

# We are interested in US states; what states have invalid numbers?
train_cc_us_state \
  .filter(train_cc_us_state.TargetValue.isNull() | (train_cc_us_state.TargetValue=="") | (train_cc_us_state.TargetValue<0)) \
  .groupBy("Province_State")\
  .count()\
  .orderBy("Province_State")\
  .show()
# Many states have invalid values, but not very often

# COMMAND ----------

# DBTITLE 1,Plots
# PLOTS 

def plot_data(data, region_lbl):
  # Convert Spark Dataframe to Pandas dataframe for plotting
  df=data.orderBy("Date").toPandas()
  
  # Get data to be plotted
  x = df["Date"]
  y = df["TargetValue"]
  y_cum = np.cumsum(df["TargetValue"])

  # Set up subplot
  fig, (ax1, ax2) = plt.subplots(2, figsize=(7,7))

  ax1.set_title('New Cases Per Day for '+region_lbl)
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Num Confirmed Cases')

  ax2.set_title('Cumulative Cases for '+region_lbl)
  ax2.set_xlabel('Date')
  ax2.set_ylabel('Num Confirmed Cases')

  ax1.plot(x, y)
  ax2.plot(x, y_cum)
  plt.tight_layout()


# COMMAND ----------

# Plot cases for US
focus_region="US"
plot_data(train_cc_country.filter(col("Country_Region")==focus_region), focus_region)

# COMMAND ----------

# Plot cases for Canada
focus_region="Canada"
plot_data(train_cc_country.filter(col("Country_Region")==focus_region), focus_region)

# COMMAND ----------

# Plot cases for New York
focus_region="New York"
plot_data(train_cc_us_state.filter(col("Province_State")==focus_region), focus_region)

# COMMAND ----------

# Plot cases for Florida
focus_region="Florida"
plot_data(train_cc_us_state.filter(col("Province_State")==focus_region), focus_region)

# COMMAND ----------

# DBTITLE 1,Check data stationarity
# EVALUATE DATA STATIONARITY

def test_stationarity(data, region_lbl):
  df = data.orderBy("Date").toPandas()
  result = adfuller(df["TargetValue"].values)
  print("Adfuller Test Results for " + region_lbl)
  print('ADF Statistic: %f' % result[0])
  print('p-value: %f' % result[1])
  print('Lags Used: %f' % result[2])
  print('Number of Observations: %f' % result[3])
  print('Critical Values')
  for k, v in result[4].items():
      print('\t%s: %.3f' % (k, v))
      
def test_autocorrelation(data, region_lbl):
  df = data.orderBy("Date").toPandas()
  X = df["TargetValue"].values
  plot_acf(X, lags=10, title="ACF for " + region_lbl)
  plot_pacf(X, lags=10, title="PACF for " + region_lbl)

def test_seasonal_decomp(data, region_lbl):
  df = data.orderBy("Date").toPandas()
  seasonal_decompose(df["TargetValue"].values, freq=7).plot()

# COMMAND ----------

# Stationarity for US data
test_stationarity(train_cc_country.filter(col("Country_Region")=="US"), "US")

# COMMAND ----------

# Autocorrelation for US data
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="US"), "US")

# COMMAND ----------

# Seasonal decomposition for US data
test_seasonal_decomp(train_cc_country.filter(col("Country_Region")=="US"), "US")

# COMMAND ----------

# Stationarity for Canada data
test_stationarity(train_cc_country.filter(col("Country_Region")=="Canada"), "Canada")

# COMMAND ----------

# Autocorrelation for Canada data
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="Canada"), "Canada")

# COMMAND ----------

# Seasonal decomposition for Canada data
test_seasonal_decomp(train_cc_country.filter(col("Country_Region")=="Canada"), "Canada")

# COMMAND ----------

# Stationarity for Florida data
test_stationarity(train_cc_us_state.filter(col("Province_State")=="Florida"), "Florida")

# COMMAND ----------

# Autocorrelation for Florida data
test_autocorrelation(train_cc_us_state.filter(col("Province_State")=="Florida"), "Florida")

# COMMAND ----------

# Seasonal decomposition for Florida data
test_seasonal_decomp(train_cc_us_state.filter(col("Province_State")=="Florida"), "Florida")

# COMMAND ----------

# DBTITLE 1,Load data (test.csv)
# LOAD DATA (test.csv)

# Note: CSVs have already been loaded using the Databricks UI
test = spark.table("test_csv")
print(test.dtypes)
test.show(10)

# For this Kaggle competition, the "test.csv" file contains a list of dates for which predictions must be generated. 
# There is 1 week overlap between test.csv and train.csv. 
# For this project, the data in train.csv will be used for the training and test data set (train/test split TBD)

# COMMAND ----------

# DBTITLE 1,Output clean data
# OUTPUT CLEAN DATA

# Download using Databricks UI
display(train_cc_country)

# COMMAND ----------

# Download using Databricks UI
display(train_cc_us_state)

# COMMAND ----------


