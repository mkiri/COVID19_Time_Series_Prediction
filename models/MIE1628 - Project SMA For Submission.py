# Databricks notebook source
# Training: 2020-02-15 to 2020-05-11
# Test: 2020-05-12 to 2020-06-10
# With SMA, the training window is a rolling window

# Countries to work with:
# - Canada
# - Japan
# - Italy
# - UK
# - Germany 
# - France

# COMMAND ----------

# DBTITLE 1,Importing Required Libraries
# MAGIC %python
# MAGIC from pyspark.sql import *
# MAGIC from pyspark.sql.functions import *
# MAGIC from pyspark.sql.window import Window
# MAGIC import pyspark.sql.functions as func

# COMMAND ----------

# DBTITLE 1,Data Exploring
display(spark.sql("SELECT * FROM g6_confirmed_cases_v2"))

# COMMAND ----------

# DBTITLE 1,SMA Functions
# MAGIC %python
# MAGIC 
# MAGIC # This can be used for the g6_confirmed_cases dataset
# MAGIC def sma(Days, Lag, Country, Prediction_Start):
# MAGIC   Country_Query = spark.sql("SELECT ROW_NUMBER() OVER(ORDER BY Date ASC) AS Row, Date, " + Country + ", lag(MA10," + str(Lag) + ") over(ORDER BY Date) AS Prediction FROM (SELECT " + Country + ", Date, avg(" + Country + ") over(ORDER BY Date ASC ROWS " + str(Days-1) + " PRECEDING) AS MA10 FROM g6_confirmed_cases_v2)")
# MAGIC   Country_Query_Prediction = Country_Query.select('Row', 'Date', Country, 'Prediction').where(Country_Query['Row'] > Prediction_Start)
# MAGIC   #display(Country_Query_Prediction)
# MAGIC   return Country_Query_Prediction
# MAGIC 
# MAGIC # This can be used for the g6_confirmed_cases dataset for plotting
# MAGIC def sma_plot(Days, Lag, Country, Prediction_Start):
# MAGIC   Country_Query = spark.sql("SELECT ROW_NUMBER() OVER(ORDER BY Date ASC) AS Row, Date, " + Country + ", lag(MA10," + str(Lag) + ") over(ORDER BY Date) AS Prediction FROM (SELECT " + Country + ", Date, avg(" + Country + ") over(ORDER BY Date ASC ROWS " + str(Days-1) + " PRECEDING) AS MA10 FROM g6_confirmed_cases_v2)")
# MAGIC   Country_Query_Prediction = Country_Query.select('Row', 'Date', Country, 'Prediction').where(Country_Query['Row'] > Prediction_Start)
# MAGIC   display(Country_Query_Prediction)
# MAGIC   return Country_Query_Prediction

# COMMAND ----------

# DBTITLE 1,SMA - 1 Day Average, Predicting Day 1 - Canada
canada_1_1 = sma_plot(1, 1, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1,SMA - 7 Day Average, Predicting Day 1 - Canada
canada_7_1 = sma_plot(7, 1, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1,SMA - 21 Day Average, Predicting Day 1 - Canada
canada_21_1 = sma_plot(21, 1, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 1 Day Average, Predicting Day 7
canada_1_7 = sma_plot(1, 7, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 7 Day Average, Predicting Day 7
canada_7_7 = sma_plot(7, 7, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 21 Day Average, Predicting Day 7
canada_21_7 = sma_plot(21, 7, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 1 Day Average, Predicting Day 21
canada_1_21 = sma_plot(1, 21, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 7 Day Average, Predicting Day 21
canada_7_21 = sma_plot(7, 21, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1, SMA - 21 Day Average, Predicting Day 21
canada_21_21 = sma_plot(21, 21, "canada_cases", 87)

# COMMAND ----------

# DBTITLE 1,SMAPE Function
# MAGIC %python
# MAGIC 
# MAGIC # Equation from -> https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# MAGIC def SMAPE(Country_Query_Prediction, Country):
# MAGIC   SMAPE = Country_Query_Prediction.withColumn("SMAPE_num", abs(col("Prediction")-col(Country)))
# MAGIC   SMAPE = SMAPE.withColumn("SMAPE_den", (abs(col("Prediction"))+abs(col(Country)))/1) #previously it was divide by 2 but the group decided this factor was not needed to use the alternative formula for SMAPE.
# MAGIC   SMAPE = SMAPE.withColumn("SMAPE_n", col("SMAPE_num")/col("SMAPE_den"))
# MAGIC   SMAPE_value = SMAPE.groupby().sum("SMAPE_n").collect()[0][0]/SMAPE.count()*100
# MAGIC   return SMAPE_value
# MAGIC 
# MAGIC def SMAPE_table(Country):
# MAGIC   # Different parameters for moving average
# MAGIC   result_1_1 = sma(1, 1, Country, 87)
# MAGIC   result_7_1 = sma(7, 1, Country, 87)
# MAGIC   result_14_1 = sma(14, 1, Country, 87)
# MAGIC   result_21_1 = sma(21, 1, Country, 87)
# MAGIC   result_1_7 = sma(1, 7, Country, 87)
# MAGIC   result_7_7 = sma(7, 7, Country, 87)
# MAGIC   result_14_7 = sma(14, 7, Country, 87)
# MAGIC   result_21_7 = sma(21, 7, Country, 87)
# MAGIC   result_1_14 = sma(1, 14, Country, 87)
# MAGIC   result_7_14 = sma(7, 14, Country, 87)
# MAGIC   result_14_14 = sma(14, 14, Country, 87)
# MAGIC   result_21_14 = sma(21, 14, Country, 87)
# MAGIC   result_1_21 = sma(1, 21, Country, 87)
# MAGIC   result_7_21 = sma(7, 21, Country, 87)
# MAGIC   result_14_21 = sma(14, 21, Country, 87)
# MAGIC   result_21_21 = sma(21, 21, Country, 87)
# MAGIC 
# MAGIC   # Creating a dataframe to make a table to summarize the results
# MAGIC   Results = Row("Model", "1 Day", "7 Day", "14 Day", "21 Day")
# MAGIC   results1 = Results("1 Day Prediction", SMAPE(result_1_1, Country), SMAPE(result_7_1, Country), SMAPE(result_14_1, Country), SMAPE(result_21_1, Country))
# MAGIC   results2 = Results("7 Day Prediction", SMAPE(result_1_7, Country), SMAPE(result_7_7, Country), SMAPE(result_14_7, Country), SMAPE(result_21_7, Country))
# MAGIC   results3 = Results("14 Day Prediction", SMAPE(result_1_14, Country), SMAPE(result_7_14, Country), SMAPE(result_14_14, Country), SMAPE(result_21_14, Country))
# MAGIC   results4 = Results("21 Day Prediction", SMAPE(result_1_21, Country), SMAPE(result_7_21, Country), SMAPE(result_14_21, Country), SMAPE(result_21_21, Country))
# MAGIC   df = spark.createDataFrame([results1, results2, results3, results4])
# MAGIC   df = df.withColumn(
# MAGIC     "1 Day", func.round(df["1 Day"], 2)).withColumn(
# MAGIC     "7 Day", func.round(df["7 Day"],2)).withColumn(
# MAGIC     "14 Day", func.round(df["14 Day"],2)).withColumn(
# MAGIC     "21 Day", func.round(df["21 Day"],2))
# MAGIC   print(Country)
# MAGIC   df.show()

# COMMAND ----------

# DBTITLE 1,RMSE Function
#RMSE Functions

def RMSE(Country_Query_Prediction, Country):
  RMSE = Country_Query_Prediction.withColumn("RMSE_num", (col("Prediction")-col(Country))**2)
  RMSE_value = ((RMSE.groupby().sum("RMSE_num").collect()[0][0]/RMSE.count())**0.5)
  return RMSE_value

def RMSE_table(Country):
  # Different parameters for moving average
  result_1_1 = sma(1, 1, Country, 87)
  result_7_1 = sma(7, 1, Country, 87)
  result_14_1 = sma(14, 1, Country, 87)
  result_21_1 = sma(21, 1, Country, 87)
  result_1_7 = sma(1, 7, Country, 87)
  result_7_7 = sma(7, 7, Country, 87)
  result_14_7 = sma(14, 7, Country, 87)
  result_21_7 = sma(21, 7, Country, 87)
  result_1_14 = sma(1, 14, Country, 87)
  result_7_14 = sma(7, 14, Country, 87)
  result_14_14 = sma(14, 14, Country, 87)
  result_21_14 = sma(21, 14, Country, 87)
  result_1_21 = sma(1, 21, Country, 87)
  result_7_21 = sma(7, 21, Country, 87)
  result_14_21 = sma(14, 21, Country, 87)
  result_21_21 = sma(21, 21, Country, 87)

  # Creating a dataframe to make a table to summarize the results
  Results = Row("Model", "1 Day", "7 Day", "14 Day", "21 Day")
  results1 = Results("1 Day Prediction", RMSE(result_1_1, Country), RMSE(result_7_1, Country), RMSE(result_14_1, Country), RMSE(result_21_1, Country))
  results2 = Results("7 Day Prediction", RMSE(result_1_7, Country), RMSE(result_7_7, Country), RMSE(result_14_7, Country), RMSE(result_21_7, Country))
  results3 = Results("14 Day Prediction", RMSE(result_1_14, Country), RMSE(result_7_14, Country), RMSE(result_14_14, Country), RMSE(result_21_14, Country))
  results4 = Results("21 Day Prediction", RMSE(result_1_21, Country), RMSE(result_7_21, Country), RMSE(result_14_21, Country), RMSE(result_21_21, Country))
  df = spark.createDataFrame([results1, results2, results3, results4])
  df = df.withColumn(
    "1 Day", func.round(df["1 Day"], 2)).withColumn(
    "7 Day", func.round(df["7 Day"],2)).withColumn(
    "14 Day", func.round(df["14 Day"],2)).withColumn(
    "21 Day", func.round(df["21 Day"],2))
  print(Country)
  df.show()


# COMMAND ----------

# R^2 Functions
# Equation from -> https://en.wikipedia.org/wiki/Coefficient_of_determination
def R2(Country_Query_Prediction, Country):
  R2 = Country_Query_Prediction
  R2_avg = R2.groupby().avg(Country).collect()[0][0]
  R2 = R2.withColumn("R2_TSS", (col(Country)-R2_avg)**2)
  R2 = R2.withColumn("R2_RSS", (col(Country)-col("Prediction"))**2)
  R2_value = (1 - (R2.groupby().sum("R2_RSS").collect()[0][0])/(R2.groupby().sum("R2_TSS").collect()[0][0]))*100 # Multiply by 100 for percentage
  return R2_value

def R2_table(Country):
  # Different parameters for moving average
  result_1_1 = sma(1, 1, Country, 87)
  result_7_1 = sma(7, 1, Country, 87)
  result_14_1 = sma(14, 1, Country, 87)
  result_21_1 = sma(21, 1, Country, 87)
  result_1_7 = sma(1, 7, Country, 87)
  result_7_7 = sma(7, 7, Country, 87)
  result_14_7 = sma(14, 7, Country, 87)
  result_21_7 = sma(21, 7, Country, 87)
  result_1_14 = sma(1, 14, Country, 87)
  result_7_14 = sma(7, 14, Country, 87)
  result_14_14 = sma(14, 14, Country, 87)
  result_21_14 = sma(21, 14, Country, 87)
  result_1_21 = sma(1, 21, Country, 87)
  result_7_21 = sma(7, 21, Country, 87)
  result_14_21 = sma(14, 21, Country, 87)
  result_21_21 = sma(21, 21, Country, 87)

  # Creating a dataframe to make a table to summarize the results
  Results = Row("Model", "1 Day", "7 Day", "14 Day", "21 Day")
  results1 = Results("1 Day Prediction", R2(result_1_1, Country), R2(result_7_1, Country), R2(result_14_1, Country), R2(result_21_1, Country))
  results2 = Results("7 Day Prediction", R2(result_1_7, Country), R2(result_7_7, Country), R2(result_14_7, Country), R2(result_21_7, Country))
  results3 = Results("14 Day Prediction", R2(result_1_14, Country), R2(result_7_14, Country), R2(result_14_14, Country), R2(result_21_14, Country))
  results4 = Results("21 Day Prediction", R2(result_1_21, Country), R2(result_7_21, Country), R2(result_14_21, Country), R2(result_21_21, Country))
  df = spark.createDataFrame([results1, results2, results3, results4])
  df = df.withColumn(
    "1 Day", func.round(df["1 Day"], 2)).withColumn(
    "7 Day", func.round(df["7 Day"],2)).withColumn(
    "14 Day", func.round(df["14 Day"],2)).withColumn(
    "21 Day", func.round(df["21 Day"],2))
  print(Country)
  df.show()


# COMMAND ----------

# DBTITLE 1,RMSE Results
RMSE_table("canada_cases")
RMSE_table("japan_cases")
RMSE_table("italy_cases")
RMSE_table("uk_cases")
RMSE_table("germany_cases")
RMSE_table("france_cases")

# COMMAND ----------

# DBTITLE 1,SMAPE Results
SMAPE_table("canada_cases")
SMAPE_table("japan_cases")
SMAPE_table("italy_cases")
SMAPE_table("UK_cases")
SMAPE_table("germany_cases")
SMAPE_table("france_cases")

# COMMAND ----------

# DBTITLE 1,R2 Results
R2_table("canada_cases")
R2_table("japan_cases")
R2_table("italy_cases")
R2_table("UK_cases")
R2_table("germany_cases")
R2_table("france_cases")
