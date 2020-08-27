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

# Note: CSVs have already been loaded using the Databricks UI
# LOAD DATA (train.csv)

# Note: CSVs have already been loaded using the Databricks UI
df_temp = spark.table("g6_all_features_csv")

data = df_temp \
  .withColumnRenamed("Date", "Date_str") \
  .withColumn("Date", to_timestamp("Date_str")) \
  .orderBy(col("Date"))

print(data.dtypes)
display(data)



# COMMAND ----------

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
    "canada_1lag","japan_1lag","italy_1lag","uk_1lag","germany_1lag","france_1lag", \
    "canada_2lag","japan_2lag","italy_2lag","uk_2lag","germany_2lag","france_2lag", \
    "canada_3lag","japan_3lag","italy_3lag","uk_3lag","germany_3lag","france_3lag", \
    "canada_ma","japan_ma","italy_ma","uk_ma","germany_ma","france_ma", \
    "canada_pt","japan_pt","italy_pt","uk_pt","germany_pt","france_pt", "dow"]

# Features extracted from additional datasets
EXTENDED_FEATURES = ORIGINAL_FEATURES + [ \
    'Canada_Outdoor_Mobility', 'Canada_indoorMobility', 'Germany_Outdoor_Mobility', 'Germany_indoorMobility', \
    'France_Outdoor_Mobility', 'France_indoorMobility', 'Japan_Outdoor_Mobility', 'Japan_indoorMobility', \
    'Italy_Outdoor_Mobility', 'Italy_indoorMobility', 'United_Kingdom_Outdoor_Mobility', 'United_Kingdom_indoorMobility', \
    'temp_canada', 'temp_uk', 'temp_italy', 'temp_japan', 'temp_germany', 'temp_france']

# COMMAND ----------

# HELPER FUNCTIONS

def train_test_split(full_data, country, feature_list, time_horizon):
  target_label=country+"_cases"
  prev=country+"_1lag"
  w = Window.orderBy(col("Date"))
  # Assemble feature vector where Canada cases are the target
  vectorAssembler = VectorAssembler(inputCols = feature_list, outputCol = 'features')
  vdata = vectorAssembler.transform(full_data) \
    .withColumnRenamed(target_label, "actual_orig") \
    .withColumn("actual", lead(col("actual_orig"), time_horizon).over(w)) \
    .withColumn("diff", col("actual_orig")-col(prev)) \
    .withColumn("actual_diff", lead(col("diff"), time_horizon).over(w)) \
    .filter(col("actual_diff").isNotNull())
  
  train = vdata.select(['features', "Date", "actual_orig", "diff", "actual", "actual_diff"]).filter(col("Date")<SPLIT_DT)
  test = vdata.select(['features', "Date", "actual_orig", "diff", "actual", "actual_diff"]).filter(col("Date")>=SPLIT_DT)

  print("Total days: " + str(vdata.count()))
  print("Total days for train dataset: " + str(train.count()))
  print("Total days for test dataset: " + str(test.count()))
  return train, test

# Train model using provided training data, parameter grid and target 
def train_model(train_data):
  # Gradient boosted regression tree model
  gbt = GBTRegressor(featuresCol = 'features', labelCol="actual_diff")
  
  # Hyperparameters to be tuned
  param_grid = (ParamGridBuilder() \
               .addGrid(gbt.maxDepth, [2, 5, 8]) \
               .addGrid(gbt.maxBins, [20, 30, 40]) \
               .addGrid(gbt.maxIter, [5, 10, 20]) \
               .build())

  # Evaluation metrics
  eval_ = RegressionEvaluator(labelCol= "actual_diff", predictionCol= "prediction", metricName="rmse")

  # Run rolling k-fold cross validation
  cv = RollingKFoldCV(estimator=gbt, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Train model
  return cv.fit(train_data)
  

# Run on test data using best trained model
def test_model(mdl):
  results_diff = mdl.transform(test)
  
  results = results_diff \
    .withColumnRenamed("prediction", "pred_diff") \
    .withColumn("prediction", col("actual")-col("actual_diff")+col("pred_diff"))
  
  # Evaluate predictions
  reg_eval = RegressionEvaluator(labelCol= "actual", predictionCol= "prediction")
  rmse = reg_eval.evaluate(results, {reg_eval.metricName: "rmse"})
  print('rmse is ' + str(rmse))
  mae = reg_eval.evaluate(results, {reg_eval.metricName: "r2"})
  print('r2 is ' + str(mae))
  
  return results 

# Calculate metrics on prediction data
def calculate_metrics(pred_data):
  return pred_data \
    .agg(\
      sqrt(avg(pow(col('prediction')-col('actual'),2))).alias("rmse"), \
      (100/count('*')*sum(abs(col('prediction')-col('actual'))/(abs(col('prediction'))+abs(col('actual'))))).alias("smape"))

# Plot results
def plot_results(data, cols, country):
  df=data.orderBy("Date").toPandas()
  x = df["Date"]
  for c in cols:
    plt.plot(x, df[c], label = c)
  plt.xlabel('Date')
  plt.ylabel('Num Confirmed Cases')
  plt.title('New Cases Per Day ('+country+')')
  plt.legend()
  plt.xticks(rotation=45)
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Canada

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Japan

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'japan'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'japan'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'japan'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'japan'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Italy

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'italy'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'italy'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'italy'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'italy'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for UK

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'uk'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'uk'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'uk'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'uk'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Germany

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'germany'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'germany'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'germany'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'germany'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for France

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'france'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'france'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'france'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'france'
train, test = train_test_split(data, target, ORIGINAL_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Number of Cases for Canada (Extended Features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, EXTENDED_FEATURES, 1)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, EXTENDED_FEATURES, 7)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, EXTENDED_FEATURES, 14)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 21 Day Prediction Horizon

# COMMAND ----------

# Get train/test data for target country
target = 'canada'
train, test = train_test_split(data, target, EXTENDED_FEATURES, 21)

train.show(5)

# COMMAND ----------

# Train model
model = train_model(train)

# Get best model with optimal hyperparameters
best_model = model.bestModel
best_params = best_model.extractParamMap()

{p[0].name: p[1] for p in best_params.items()}

# COMMAND ----------

# Get predictions using test data
pred = test_model(model)
pred.show(10)

# COMMAND ----------

plot_results(pred, ["actual", "prediction"], target)

# COMMAND ----------

# Evaluate performance/accuracy
pred_metrics = calculate_metrics(pred)
pred_metrics.show()

# COMMAND ----------

# MAGIC %md
