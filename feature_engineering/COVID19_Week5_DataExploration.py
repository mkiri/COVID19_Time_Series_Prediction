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

# SELECTING G7 COUNTRIES (excluding US)

# Get absolute value of cases for each country
canada_cc = train_cc_country.filter(col("Country_Region")=="Canada").select(col("Date"), abs(col("TargetValue")).alias("canada_cases"))
japan_cc = train_cc_country.filter(col("Country_Region")=="Japan").select(col("Date"), abs(col("TargetValue")).alias("japan_cases"))
italy_cc = train_cc_country.filter(col("Country_Region")=="Italy").select(col("Date"), abs(col("TargetValue")).alias("italy_cases"))
uk_cc = train_cc_country.filter(col("Country_Region")=="United Kingdom").select(col("Date"), abs(col("TargetValue")).alias("uk_cases"))
ger_cc =train_cc_country.filter(col("Country_Region")=="Germany").select(col("Date"), abs(col("TargetValue")).alias("germany_cases"))
france_cc = train_cc_country.filter(col("Country_Region")=="France").select(col("Date"), abs(col("TargetValue")).alias("france_cases"))


# COMMAND ----------

# PLOTTING FUNCTIONS

# Plot daily cases for G6 countries
def plot_multi_daily_cases(data, cols):
  df=data.orderBy("Date").toPandas()
  x = df["Date"]
  for c in cols:
    plt.plot(x, df[c], label = c)
  plt.xlabel('Date')
  plt.ylabel('Num Confirmed Cases')
  plt.title('New Cases Per Day')
  plt.legend()
  plt.show()

# Plot cumulative cases for G6 countries
def plot_multi_cum_cases(data, cols):
  df=data.orderBy("Date").toPandas()
  x = df["Date"]
  for c in cols:
    plt.plot(x, np.cumsum(df[c]), label = c)
  plt.xlabel('Date')
  plt.ylabel('Cumulative Cases')
  plt.title('New Cases Per Day')
  plt.legend()
  plt.show()

# COMMAND ----------

# Remove outliers (values above given percentile of data)
def remove_outliers(data, max_percentile, col_name):
  # Get value at specified percentile
  upperBound = data.approxQuantile(col_name, [max_percentile], 0)[0]
  return data \
    .withColumn(col_name+"_filt", when(col(col_name)>upperBound, lit(upperBound)).otherwise(col(col_name))) \
    .select(col("Date"), col(col_name).alias(col_name+"_orig"), col(col_name+"_filt").alias(col_name))

# COMMAND ----------

# Target percentile
target_pct = 0.97

# Removed values greater than target percentile for canada data
canada_cc_filt = remove_outliers(canada_cc, target_pct, "canada_cases")
plot_multi_daily_cases(canada_cc_filt,["canada_cases_orig", "canada_cases"])

# COMMAND ----------

# Removed values greater than target percentile for japan data
japan_cc_filt = remove_outliers(japan_cc, target_pct, "japan_cases")
plot_multi_daily_cases(japan_cc_filt,["japan_cases_orig", "japan_cases"])

# COMMAND ----------

# Removed values greater than target percentile for italy data
italy_cc_filt = remove_outliers(italy_cc, target_pct, "italy_cases")
plot_multi_daily_cases(italy_cc_filt,["italy_cases_orig", "italy_cases"])

# COMMAND ----------

# Removed values greater than target percentile for uk data
uk_cc_filt = remove_outliers(uk_cc, target_pct, "uk_cases")
plot_multi_daily_cases(uk_cc_filt,["uk_cases_orig", "uk_cases"])

# COMMAND ----------

# Removed values greater than target percentile for germany data
germany_cc_filt = remove_outliers(ger_cc, target_pct, "germany_cases")
plot_multi_daily_cases(germany_cc_filt,["germany_cases_orig", "germany_cases"])

# COMMAND ----------

# Removed values greater than target percentile for france data
france_cc_filt = remove_outliers(france_cc, target_pct, "france_cases")
plot_multi_daily_cases(france_cc_filt,["france_cases_orig", "france_cases"])

# COMMAND ----------

# Rename columns for easy join
canada_cc = canada_cc_filt.select(col("Date"), "canada_cases")
japan_cc = japan_cc_filt.select(col("Date").alias("japan_dt"), "japan_cases")
italy_cc = italy_cc_filt.select(col("Date").alias("italy_dt"), "italy_cases")
uk_cc = uk_cc_filt.select(col("Date").alias("uk_dt"), "uk_cases")
germany_cc = germany_cc_filt.select(col("Date").alias("germany_dt"), "germany_cases")
france_cc = france_cc_filt.select(col("Date").alias("france_dt"), "france_cases")

# Build G6 dataset
g6_cc = canada_cc \
  .join(japan_cc, canada_cc.Date==japan_cc.japan_dt) \
  .join(italy_cc, canada_cc.Date==italy_cc.italy_dt) \
  .join(uk_cc, canada_cc.Date==uk_cc.uk_dt) \
  .join(germany_cc, canada_cc.Date==germany_cc.germany_dt) \
  .join(france_cc, canada_cc.Date==france_cc.france_dt) \
  .select(canada_cc.Date, col("canada_cases"), col("japan_cases"), col("italy_cases"), col("uk_cases"), col("germany_cases"), col("france_cases"))
  
g6_cc.show()

# COMMAND ----------

# Plot daily cases for G6 countries
plot_multi_daily_cases(g6_cc,list(g6_cc.schema.names)[1:])

# COMMAND ----------

# Plot cumulative cases for G6 countries
plot_multi_cum_cases(g6_cc,list(g6_cc.schema.names)[1:])

# COMMAND ----------

# EVALUATE STATIONARITY OF G6 COUNTRIES

# Adfuller Test
df=g6_cc.orderBy("Date").toPandas()
all_cc = []
for c in df.columns[1:]: 
  result = adfuller(df[c].values)
  all_cc.append([c.split('_')[0]]+list(result[0:4]))
  
print(pd.DataFrame(all_cc, columns = ["Country","ADF Stat","p-value","Lags Used","Observations"]))

# COMMAND ----------

# Test autocorrelation for Canada
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="Canada"), "Canada")
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="Japan"), "Japan")
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="Italy"), "Italy")
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="United Kingdom"), "UK")
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="Germany"), "Germany")
test_autocorrelation(train_cc_country.filter(col("Country_Region")=="France"), "France")

# COMMAND ----------

# ADDING MOVING AVERAGE AS A FEATURE
days = lambda i: i * 86400 
roll_window = (Window.orderBy(col("Date").cast("long")).rangeBetween(-days(6), 0))
g6_cc_ma = g6_cc \
  .withColumn('canada_ma', avg("canada_cases").over(roll_window)) \
  .withColumn('japan_ma', avg("japan_cases").over(roll_window)) \
  .withColumn('italy_ma', avg("italy_cases").over(roll_window)) \
  .withColumn('uk_ma', avg("uk_cases").over(roll_window)) \
  .withColumn('germany_ma', avg("germany_cases").over(roll_window)) \
  .withColumn('france_ma', avg("france_cases").over(roll_window)) \
  .orderBy(col("Date"))

g6_cc_ma.show()

# COMMAND ----------

# Plot daily cases vs moving average for Canada
plot_multi_daily_cases(g6_cc_ma,["canada_cases", "canada_ma"])

# COMMAND ----------

# Plot daily cases vs moving average for Japan
plot_multi_daily_cases(g6_cc_ma,["japan_cases", "japan_ma"])

# COMMAND ----------

# Plot daily cases vs moving average for Italy
plot_multi_daily_cases(g6_cc_ma,["italy_cases", "italy_ma"])

# COMMAND ----------

# Plot daily cases vs moving average for UK
plot_multi_daily_cases(g6_cc_ma,["uk_cases", "uk_ma"])

# COMMAND ----------

# Plot daily cases vs moving average for Germany
plot_multi_daily_cases(g6_cc_ma,["germany_cases", "germany_ma"])

# COMMAND ----------

# Plot daily cases vs moving average for France
plot_multi_daily_cases(g6_cc_ma,["france_cases", "france_ma"])

# COMMAND ----------

# EXPLORING POWER TRANSFORM

def power_transform_feature(data, col):
  df=data.orderBy("Date").toPandas()
  df['pwr_tf']=df[col]
  
  sc = MinMaxScaler(feature_range=(1, 2))
  pt = PowerTransformer(method='box-cox')
  pipeline = Pipeline(steps=[('s', sc),('p', pt)])
  df[['pwr_tf']] = pipeline.fit_transform(df[['pwr_tf']])

  fig, axs = plt.subplots(2, 2, figsize=(10,7))
  fig.suptitle(col)

  axs[0, 0].plot(df["Date"], df[col])
  axs[0, 0].set(xlabel='Date', ylabel='Num Cases')
  axs[0, 0].set_title('Daily Cases')

  axs[0, 1].hist(df[col])
  axs[0, 1].set(xlabel='Num Cases', ylabel='Freq')
  axs[0, 1].set_title('Histogram Daily Cases')

  axs[1, 0].plot(df["Date"], df['pwr_tf'])
  axs[1, 0].set(xlabel='Date', ylabel='Num Cases')
  axs[1, 0].set_title('After Power Transform')

  axs[1, 1].hist(df['pwr_tf'])
  axs[1, 1].set(xlabel='Num Cases', ylabel='Freq')
  axs[1, 1].set_title('Histogram After PT')

  for ax in axs.flat:
      ax.set(xlabel='Date', ylabel='Num Cases')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
      ax.label_outer()

  plt.tight_layout()


# COMMAND ----------

power_transform_feature(g6_cc, "canada_cases")

# COMMAND ----------

power_transform_feature(g6_cc, "japan_cases")

# COMMAND ----------

power_transform_feature(g6_cc, "italy_cases")

# COMMAND ----------

power_transform_feature(g6_cc, "uk_cases")

# COMMAND ----------

power_transform_feature(g6_cc, "germany_cases")

# COMMAND ----------

power_transform_feature(g6_cc, "france_cases")

# COMMAND ----------

# ADDING POWER TRANSFORM (BOX-COX) AS A FEATURE

# Prep data
df=g6_cc_ma.orderBy("Date").toPandas()
df['canada_pt']=df['canada_cases']
df['japan_pt']=df['japan_cases']
df['italy_pt']=df['italy_cases']
df['uk_pt']=df['uk_cases']
df['germany_pt']=df['germany_cases']
df['france_pt']=df['france_cases']

# Initialize pipeline
sc = MinMaxScaler(feature_range=(1, 2))
pt = PowerTransformer(method='box-cox')
pipeline = Pipeline(steps=[('s', sc),('p', pt)])

# Apply transform
col_list = ['canada_pt','japan_pt','italy_pt','uk_pt','germany_pt','france_pt']
df[col_list] = pipeline.fit_transform(df[col_list])

g6_pt = spark.createDataFrame(df)

# COMMAND ----------

# ADD DAY OF WEEK (INTEGER) AS FEATURE

g6_features = g6_pt.withColumn('dow', dayofweek(col('Date')))
g6_features.show()


# COMMAND ----------

display(g6_features)

# COMMAND ----------


