# Databricks notebook source
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.regression import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import datetime


# COMMAND ----------

import numpy as np
from multiprocessing.pool import ThreadPool
from pyspark.ml.tuning import _parallelFitTasks


class RollingKFoldCV(CrossValidator):
    """
    Modified CrossValidator to allow rolling k-fold cross validation using an increasing window. 
    Aimed to prevent data leakage associated with time series data (can't predict past using future data).
    """
        
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        metrics = [0.0] * numModels

        # Use rolling window instead of random folds
        rowNumCol = self.uid + "_rownum"
        w = Window().orderBy(lit('A')) # Dummy window to create row number
        df = dataset.select("*", row_number().over(w).alias(rowNumCol))
        h = df.count()/(nFolds+1)

        pool = ThreadPool(processes=self.getParallelism())
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        for i in range(nFolds):
            # Get rolling (increasing) window
            validateLB = (i + 1) * h
            validateUB = (i + 2) * h
            validation = df.filter((df[rowNumCol] >= validateLB) & (df[rowNumCol] < validateUB)).cache()
            train = df.filter(df[rowNumCol] < validateLB).cache()

            tasks = _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels))

# COMMAND ----------

# LOAD DATA (train.csv)

# Note: CSVs have already been loaded using the Databricks UI
df_temp = spark.table("g6_features_csv")

data = df_temp \
  .withColumn("Date2", to_timestamp("Date")) \
  .select( \
    col("Date2").alias("Date"), \
    "canada_cases","japan_cases","italy_cases","uk_cases","germany_cases","france_cases", \
    "canada_ma","japan_ma","italy_ma","uk_ma","germany_ma","france_ma", \
    "canada_pt","japan_pt","italy_pt","uk_pt","germany_pt","france_pt" \
  ) \
  .filter(col('Date')>='2020-02-15') \
  .orderBy(col("Date"))


print(data.dtypes)
data.show(10)



# COMMAND ----------

# from pyspark.sql.functions import *
# df = spark.table("HIVE_DB.HIVE_TABLE")
data.agg(min(col("Date")), max(col("Date"))).show()

# COMMAND ----------

# CONSTANTS

# % of data used for training vs testing
TRAIN_TEST_SPLIT = 0.75

# First date in feature data
FIRST_DT = '2020-02-15'
# FIRST_DT = '2020-01-23'

# Determine date used to split data into train/test
v = data.count()*TRAIN_TEST_SPLIT
SPLIT_DT = (datetime.datetime.strptime(FIRST_DT, "%Y-%m-%d") + datetime.timedelta(days=v)).strftime("%Y-%m-%d")
print(SPLIT_DT)

# Features extracted from original COVID data
ORIGINAL_FEATURES = [ \
    "canada_cases","japan_cases","italy_cases","uk_cases","germany_cases","france_cases", \
    "canada_ma","japan_ma","italy_ma","uk_ma","germany_ma","france_ma", \
    "canada_pt","japan_pt","italy_pt","uk_pt","germany_pt","france_pt"]

# Features extracted from additional datasets
EXTENDED_FEATURES = ORIGINAL_FEATURES + [ \
    "canada_cases","japan_cases","italy_cases","uk_cases","germany_cases","france_cases", \
    "canada_ma","japan_ma","italy_ma","uk_ma","germany_ma","france_ma", \
    "canada_pt","japan_pt","italy_pt","uk_pt","germany_pt","france_pt"]

# COMMAND ----------

# HELPER FUNCTIONS

# Train model using provided training data, parameter grid and target 
def train_model(train_data, target):
  # Gradient boosted regression tree model
  gbt = GBTRegressor(featuresCol = 'features', labelCol=target)
  
  # Hyperparameters to be tuned
  param_grid = (ParamGridBuilder() \
               .addGrid(gbt.maxDepth, [5, 8]) \
               .addGrid(gbt.maxBins, [20, 30, 40]) \
               .addGrid(gbt.maxIter, [5, 10, 20]) \
               .build())

  # Evaluation metrics
  eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="rmse")

  # Run rolling k-fold cross validation
  cv = RollingKFoldCV(estimator=gbt, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Train model
  return cv.fit(train_data)
  

# Run on test data using best trained model
def test_model(mdl, target):
  results = mdl.transform(test)
  
  # Evaluate predictions
  reg_eval = RegressionEvaluator(labelCol= target, predictionCol= "prediction")
  rmse = reg_eval.evaluate(results, {reg_eval.metricName: "rmse"})
  print('rmse is ' + str(rmse))
  mae = reg_eval.evaluate(results, {reg_eval.metricName: "mae"})
  print('mae is ' + str(mae))
  
  # Dummy window
  w = Window().orderBy(lit('A'))
  # Get date column
  return results \
    .select(col('prediction'), col(target).alias("actual"), 'features') \
    .withColumn('day_num', row_number().over(w)) \
    .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))

# Calculate metrics on prediction data
def calculate_metrics(pred_data):
  w = Window.orderBy(col("Date")).rowsBetween(Window.unboundedPreceding, Window.currentRow)
  return pred_data \
    .withColumn("rmse", sqrt(avg(pow(col('prediction')-col('actual'),2)).over(w))) \
    .withColumn("smape", 100/col('day_num')*sum(abs(col('prediction')-col('actual'))/(abs(col('prediction'))+abs(col('actual')))).over(w))

# Plot results
def plot_results(data, cols, label):
  df=data.orderBy("Date").toPandas()
  x = df["Date"]
  for c in cols:
    plt.plot(x, df[c], label = c)
  plt.xlabel('Date')
  plt.ylabel('Num Confirmed Cases')
  plt.title('New Cases Per Day ('+label+')')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Canada

# COMMAND ----------

target = 'canada_cases'

# Assemble feature vector where Canada cases are the target
vectorAssembler = VectorAssembler(inputCols = ORIGINAL_FEATURES, outputCol = 'features')
vdata = vectorAssembler.transform(data)
train = vdata.select(['features', target]).filter(col("Date")<SPLIT_DT)
test = vdata.select(['features', target]).filter(col("Date")>=SPLIT_DT)

print("Total days: " + str(vdata.count()))
print("Total days for train dataset: " + str(train.count()))
print("Total days for test dataset: " + str(test.count()))

# vdata.show(3)

# COMMAND ----------

# Train model
model = train_model(train, target)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model, target)

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)

# RMSE and sMAPE values for 1 day, 7 day, 14 day and 30 day predictions
pred_metrics.select("day_num","rmse","smape").filter(col("day_num").isin(1,7,14,30)).show()

# COMMAND ----------


