# Databricks notebook source
# MAGIC %md # Importing Libraries

# COMMAND ----------

## importing libraries to be used in the linear regression model development process
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.regression import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

import numpy as np
from multiprocessing.pool import ThreadPool
from pyspark.ml.tuning import _parallelFitTasks

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import datetime


# COMMAND ----------

# MAGIC %md # K Fold Cross Validation

# COMMAND ----------

# k fold cross validation function created by Menaka. Used same cross validation funcion to ensure consistency across all ML models.
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
            
        bestIndex = np.argmax(metrics)

        #if eva.isLargerBetter():
            #bestIndex = np.argmax(metrics)
        ##else:
            #bestIndex = np.argmin(metrics)
        
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics, subModels))

# COMMAND ----------

# MAGIC %md #SMAPE and RMSE functions

# COMMAND ----------

import numpy as np
#creating function to calculate SMAPE. Function to be called on after model is ran on test set.
def smape(A, F):
    return (100/len(A)) * np.sum(np.abs(F - A) / ((np.abs(A) + np.abs(F))))
  
#creating function to calculate RMSE. Function to be called on after model is ran on test set.
def rmse(A,F):
    return np.sqrt(np.sum((F-A)**2)/len(A))


# COMMAND ----------

# MAGIC %md # Importing table as DF: Lin Reg

# COMMAND ----------

#creating spark dataframe from table of complete feature set

#df = spark.table('features_new_csv')
df = spark.table('g6_all_features_csv')

# importing double type as features were imported as string
from pyspark.sql.types import DoubleType

# counting row totals for test/train split 
df.count()

# COMMAND ----------

# MAGIC %md # Vectorization and Feature Prep: Lin Reg 1 Day Window

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc
import pyspark.sql.functions as func
from pyspark.sql.functions import percent_rank


# creating a list of column names using the original dataframe columns
#first creating new dataframe to work on
column_info = df
#dropping date column as it will not be needed for now
columns_to_drop = ['Date' ]
column_info = column_info.drop(*columns_to_drop)
#creating a list of column names
column_list=column_info.schema.names
#viewing list of column names
#column_info.schema.names

#creating copy of original dataframe to modify
df_new = df
# changing string datatype to double for every column
for c in df.columns:
    # add condition for the cols to be type cast
    df_new=df_new.withColumn(c, df[c].cast('double'))

# Creating lag for target data from features with specified window being 1,7,14 or 21. Done using lead function. Additionally, changing date datatype from double back to a timestamp.
df_new = df_new \
  .withColumn("Date", to_timestamp("Date")) 
df_lag = df_new.withColumn('canada_lead',
                        func.lead(df_new['canada_cases'],1)
                                 .over(Window.orderBy("Date")))

#First Vectorizing (creating a column with all feature data in one). Next , Splitting the data into training data and testing data using a 75/25 split. 
#using canada_lead as target
target = 'canada_lead'

# Creating vectors of feature columns
vectorAssembler = VectorAssembler(inputCols = column_list, outputCol = 'features')
vectors = vectorAssembler.transform(df_lag)

#split into train and test, used 75/25 split. 
w = Window().partitionBy(lit('a')).orderBy(lit('a'))
df_final = vectors.withColumn("row_num", row_number().over(w)/df.count())
train = df_final.where("row_num <= .75").drop("row_num").select(['features',target])
test = df_final.where("row_num > .75").drop("row_num").select(['features',target]).filter(df_final.canada_lead. isNotNull())


display(train)

# COMMAND ----------

# MAGIC %md # Model Development and Hyperparamter Tuning: Lin Reg 1 Day Window

# COMMAND ----------

# Modelling and Hyperparamter tuning. Below a lienar regression model is initialized, trained, and tuned using a variety of different hyperparameters. 
 
#Initializing linear regression model. The feature column is "features", which was created through vectorization of all of the non target features. labelCol is the target feature, which in this case is the lag of Canada cases by the specified window.
lr = LinearRegression(featuresCol = 'features', labelCol=target)
  
#hyperparameter tuning: creating a list of hyperparameters for the grid search
#---------------------------------------------------------------------------------
#maxIterations refers to the number of iterations used in order to train the algorithm

# elastic net parameters refers to the penalty used in regularization. L2 penalty is used for a value of 0, L1 penalty is used for a value of 1. A combination of L1 and L2 is used for a value of 0.5. These were tested initially, however L2 and none are the only penalty availble when using the Huber Loss function.

#Huber loss is a option for loss function that makes the regression more robust, in the sense that it is much less sensitive to outliers than the Squared Error Loss Function. The Squared Error Loss Function is the sum of squared distances between actual and predicted values  .As the residuals will later show. There are outliers in this data which makes this loss function optimal over squared error loss. Could not run param grid with both as Huber does not allow for elasticNetParam but tested both and Huber was much better.

# Epsilon can only be used with the Huber loss function and is the shape parameter to control the amount of robustness. Through trial and error 4.1 was found as optimal.

# the regularization paramter used in linear regression. A tuning parameter used to control the impact on bias and variance.
#----------------------------------------------------------------------------------------------
param_grid = (ParamGridBuilder() \
               .addGrid(lr.maxIter, [ 5,10,25]) \
               .addGrid(lr.elasticNetParam, [0.0])   \
               .addGrid(lr.regParam, [0.01, 0.1])   \
               .addGrid(lr.loss,["huber"]) \
               .addGrid(lr.epsilon,[4.1]) \
               .addGrid(lr.fitIntercept, [True,False]) \
               .addGrid(lr.standardization, [True,False]) \
               .build())
           
  # Setting the evaluation parameters for the linear regression model. R^2 is defined as the proportion of the variance in the dependent variable that is predictable from the independent. This means that it represents the squared correlation between the predicted and actual cases. The best model will have the highest R2 , closest to 1.0 .
eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="r2")

  # Run rolling k-fold cross validation, provided by Menaka. Using param_grid and r2 as the evaluation metric. 
  
cv = RollingKFoldCV(estimator=lr, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Training model... returns model with optimal hyperparameters that have the best (highest) r2 value.
cvModel = cv.fit(train)

# storing best paramters to be viewed later
bestModel = cvModel.bestModel
bestParams = bestModel.extractParamMap()


# COMMAND ----------

# MAGIC %md # Residual Plot: Lin Reg 1 Day Window

# COMMAND ----------

# residual plot using built in residual plot options, shows good results but decent amount of outliers. Easier to see when Plot Options are selected as databricks has poor graphing support.
display(bestModel,train)

# COMMAND ----------

# MAGIC %md # Best Hyperparameters: Lin Reg 1 Day Window

# COMMAND ----------

#observing best hyperparameters to get a better understanding of the model. This process was used and repeated for tuning as sometimes discluding parameters had a positive impact on r2 and SMAPE.

'Best Param (regParam): ', bestModel._java_obj.getRegParam(), 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam(), 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter(), 'Best Param (solver): ', bestModel._java_obj.getSolver(), 'Best Param (epsilon): ', bestModel._java_obj.getEpsilon(),  bestModel._java_obj.getStandardization(), 'Best Param (standardization): ', bestModel._java_obj.getFitIntercept(), 'Best Param (fitIntercept): '

# COMMAND ----------

# MAGIC %md # Fitting Model on Testing Data: Lin Reg 1 Day Window

# COMMAND ----------

from pyspark.sql.functions import when
# Fitting to testing data using model with optimal hyperparamters found previously. 
results = cvModel.transform(test)
results_train = cvModel.transform(train)
  
# Initializing r2 as eval metric. Next evaluating predictions using r2 as the evaluation metric. 
reg_eval = RegressionEvaluator(labelCol= target, predictionCol= "prediction")
r2_test = reg_eval.evaluate(results, {reg_eval.metricName: "r2"})
r2_train = reg_eval.evaluate(results_train, {reg_eval.metricName: "r2"})


 #Creating a dummy window to be able to order features by date
w = Window().orderBy(lit('A'))
#Get date column as previous one was deleted before vectorization
results = results \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))
results_train = results_train \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))


# removing all predictions with negative case prediction as this is impossible (cant be less than 0 cases). Used 1 case instead of 0 to not mess with the eval metrics that are susceptable to 0s.
pred = results.withColumn("prediction", when(results["prediction"] < 0, 1).otherwise(results["prediction"]))
pred_train = results_train.withColumn("prediction", when(results_train["prediction"] < 0, 1).otherwise(results_train["prediction"]))

# creating a numpy array of actual canada cases (target) as it is easier to perform arithmatic on. Denoted as A. Same done with predictions denoted as F.
A = np.array(pred.select('canada_actual').collect())
F = np.array(pred.select('prediction').collect())
A_train = np.array(pred_train.select('canada_actual').collect())
F_train = np.array(pred_train.select('prediction').collect())

#printing out R2 value and RMSE value of test data. Comparing r2 for train and test to find if data is overfitting , underfitting, or a good fit.
print('r2 test:' + str(r2_test))
print('r2 train:' + str(r2_train))
print('SMAPE test:')
print(smape(A, F))
print('SMAPE train:')
print(smape(A_train, F_train))
 #looking at the values it is clear the model is overfitting on the training data, with low bias and high variance. This is supported by a high r2 score in training of 0.9 and a poor r2 in testing of 0.65. The SMAPE is satisfactory as 6.67% for test data.

# COMMAND ----------

# MAGIC  %md # Plotting Actual values vs Predicted values: Lin Reg 1 Day Window

# COMMAND ----------

#Observing graph of predictions vs actual for target "canada_cases". It might be neccesary to click Plot Options if Databricks doesnt save settings.
#testing data
# select line chart with canada_actual and prediction under values and date under key

display(pred)

# COMMAND ----------

#training data
# select line chart with canada_actual and prediction under values and date under key
# from this graph it can be seen that the data is slightly overfitting on the training data. This is further supported by the high R2 number of 0.9 for train and lower number off 0.65 for test. This means that the model has high variance and low bias (overfitting). 
display(pred_train)

# COMMAND ----------

# MAGIC %md #Vectorization and Feature Prep: Lin Reg 7 Day Window

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc
import pyspark.sql.functions as func
from pyspark.sql.functions import percent_rank


# creating a list of column names using the original dataframe columns
#first creating new dataframe to work on
column_info = df
#dropping date column as it will not be needed for now
columns_to_drop = ['Date', ]
column_info = column_info.drop(*columns_to_drop)
#creating a list of column names
column_list=column_info.schema.names
#viewing list of column names
#column_info.schema.names

#creating copy of original dataframe to modify
df_new = df
# changing string datatype to double for every column
for c in df.columns:
    # add condition for the cols to be type cast
    df_new=df_new.withColumn(c, df[c].cast('double'))

# Creating lag for target data from features with specified window being 1,7,14 or 21. Done using lead function. Additionally, changing date datatype from double back to a timestamp.
df_new = df_new \
  .withColumn("Date", to_timestamp("Date")) 
df_lag = df_new.withColumn('canada_lead',
                        func.lead(df_new['canada_cases'],7)
                                 .over(Window.orderBy("Date")))

#First Vectorizing (creating a column with all feature data in one). Next , Splitting the data into training data and testing data using a 75/25 split. 
#using canada_lead as target
target = 'canada_lead'

# Creating vectors of feature columns
vectorAssembler = VectorAssembler(inputCols = column_list, outputCol = 'features')
vectors = vectorAssembler.transform(df_lag)

#split into train and test, used 75/25 split. 
w = Window().partitionBy(lit('a')).orderBy(lit('a'))
df_final = vectors.withColumn("row_num", row_number().over(w)/df.count())
train = df_final.where("row_num <= .75").drop("row_num").select(['features',target])
test = df_final.where("row_num > .75").drop("row_num").select(['features',target]).filter(df_final.canada_lead. isNotNull())


display(train)

# COMMAND ----------

# MAGIC %md #Model Development and Hyperparamter Tuning: Lin Reg 7 Day Window

# COMMAND ----------

# Modelling and Hyperparamter tuning. Below a lienar regression model is initialized, trained, and tuned using a variety of different hyperparameters. 
 
#Initializing linear regression model. The feature column is "features", which was created through vectorization of all of the non target features. labelCol is the target feature, which in this case is the lag of Canada cases by the specified window.
lr = LinearRegression(featuresCol = 'features', labelCol=target)
  
#hyperparameter tuning: creating a list of hyperparameters for the grid search
#---------------------------------------------------------------------------------
#maxIterations refers to the number of iterations used in order to train the algorithm

# elastic net parameters refers to the penalty used in regularization. L2 penalty is used for a value of 0, L1 penalty is used for a value of 1. A combination of L1 and L2 is used for a value of 0.5. These were tested initially, however L2 and none are the only penalty availble when using the Huber Loss function.

#Huber loss is a option for loss function that makes the regression more robust, in the sense that it is much less sensitive to outliers than the Squared Error Loss Function. The Squared Error Loss Function is the sum of squared distances between actual and predicted values  .As the residuals will later show. There are outliers in this data which makes this loss function optimal over squared error loss. Could not run param grid with both as Huber does not allow for elasticNetParam but tested both and Huber was much better.

# Epsilon can only be used with the Huber loss function and is the shape parameter to control the amount of robustness. Through trial and error 4.1 was found as optimal.

# the regularization paramter used in linear regression. A tuning parameter used to control the impact on bias and variance.
#----------------------------------------------------------------------------------------------
param_grid = (ParamGridBuilder() \
               .addGrid(lr.maxIter, [ 5,10,25]) \
               .addGrid(lr.elasticNetParam, [0.0])   \
               .addGrid(lr.regParam, [0.01, 0.1])   \
               .addGrid(lr.loss,["huber"]) \
               .addGrid(lr.epsilon,[4.1]) \
               .addGrid(lr.fitIntercept, [True,False]) \
               .addGrid(lr.standardization, [True,False]) \
               .build())
           
  # Setting the evaluation parameters for the linear regression model. R^2 is defined as the proportion of the variance in the dependent variable that is predictable from the independent. This means that it represents the squared correlation between the predicted and actual cases. The best model will have the highest R2 , closest to 1.0 .
eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="r2")

  # Run rolling k-fold cross validation, provided by Menaka. Using param_grid and r2 as the evaluation metric. 
  
cv = RollingKFoldCV(estimator=lr, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Training model... returns model with optimal hyperparameters that have the best (highest) r2 value.
cvModel = cv.fit(train)

# storing best paramters to be viewed later
bestModel = cvModel.bestModel
bestParams = bestModel.extractParamMap()


# COMMAND ----------

# MAGIC %md #Residual Plot: Lin Reg 7 Day Window

# COMMAND ----------

# residual plot using built in residual plot options, shows good results but decent amount of outliers. Easier to see when Plot Options are selected as databricks has poor graphing support.
display(bestModel,train)

# COMMAND ----------

# MAGIC %md #Best Hyperparameters: Lin Reg 7 Day Window

# COMMAND ----------

#observing best hyperparameters to get a better understanding of the model. This process was used and repeated for tuning as sometimes discluding parameters had a positive impact on r2 and SMAPE.

'Best Param (regParam): ', bestModel._java_obj.getRegParam(), 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam(), 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter(), 'Best Param (solver): ', bestModel._java_obj.getSolver(), 'Best Param (epsilon): ', bestModel._java_obj.getEpsilon(),  bestModel._java_obj.getStandardization(), 'Best Param (standardization): ', bestModel._java_obj.getFitIntercept(), 'Best Param (fitIntercept): '

# COMMAND ----------

# MAGIC %md #Fitting Model on Testing Data: Lin Reg 7 Day Window

# COMMAND ----------

from pyspark.sql.functions import when
# Fitting to testing data using model with optimal hyperparamters found previously. 
results = cvModel.transform(test)
results_train = cvModel.transform(train)
  
# Initializing r2 as eval metric. Next evaluating predictions using r2 as the evaluation metric. 
reg_eval = RegressionEvaluator(labelCol= target, predictionCol= "prediction")
r2_test = reg_eval.evaluate(results, {reg_eval.metricName: "r2"})
r2_train = reg_eval.evaluate(results_train, {reg_eval.metricName: "r2"})


 #Creating a dummy window to be able to order features by date
w = Window().orderBy(lit('A'))
#Get date column as previous one was deleted before vectorization
results = results \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))
results_train = results_train \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))


# removing all predictions with negative case prediction as this is impossible (cant be less than 0 cases). Used 1 case instead of 0 to not mess with the eval metrics that are susceptable to 0s.
pred = results.withColumn("prediction", when(results["prediction"] < 0, 1).otherwise(results["prediction"]))
pred_train = results_train.withColumn("prediction", when(results_train["prediction"] < 0, 1).otherwise(results_train["prediction"]))

# creating a numpy array of actual canada cases (target) as it is easier to perform arithmatic on. Denoted as A. Same done with predictions denoted as F.
A = np.array(pred.select('canada_actual').collect())
F = np.array(pred.select('prediction').collect())
A_train = np.array(pred_train.select('canada_actual').collect())
F_train = np.array(pred_train.select('prediction').collect())

#printing out R2 value and RMSE value of test data. Comparing r2 for train and test to find if data is overfitting , underfitting, or a good fit.
print('r2 test:' + str(r2_test))
print('r2 train:' + str(r2_train))
print('SMAPE test:')
print(smape(A, F))
print('SMAPE train:')
print(smape(A_train, F_train))
 

# COMMAND ----------

# MAGIC %md #Plotting Actual values vs Predicted values: Lin Reg 7 Day Window

# COMMAND ----------

#Observing graph of predictions vs actual for target "canada_cases". It might be neccesary to click Plot Options if Databricks doesnt save settings.
#testing data
# select line chart with canada_actual and prediction under values and date under key

display(pred)

# COMMAND ----------

#training data
# select line chart with canada_actual and prediction under values and date under key

display(pred_train)

# COMMAND ----------

# MAGIC %md #Vectorization and Feature Prep: Lin Reg 14 Day Window

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc
import pyspark.sql.functions as func
from pyspark.sql.functions import percent_rank


# creating a list of column names using the original dataframe columns
#first creating new dataframe to work on
column_info = df
#dropping date column as it will not be needed for now
columns_to_drop = ['Date', ]
column_info = column_info.drop(*columns_to_drop)
#creating a list of column names
column_list=column_info.schema.names
#viewing list of column names
#column_info.schema.names

#creating copy of original dataframe to modify
df_new = df
# changing string datatype to double for every column
for c in df.columns:
    # add condition for the cols to be type cast
    df_new=df_new.withColumn(c, df[c].cast('double'))

# Creating lag for target data from features with specified window being 1,7,14 or 21. Done using lead function. Additionally, changing date datatype from double back to a timestamp.
df_new = df_new \
  .withColumn("Date", to_timestamp("Date")) 
df_lag = df_new.withColumn('canada_lead',
                        func.lead(df_new['canada_cases'],14)
                                 .over(Window.orderBy("Date")))

#First Vectorizing (creating a column with all feature data in one). Next , Splitting the data into training data and testing data using a 75/25 split. 
#using canada_lead as target
target = 'canada_lead'

# Creating vectors of feature columns
vectorAssembler = VectorAssembler(inputCols = column_list, outputCol = 'features')
vectors = vectorAssembler.transform(df_lag)

#split into train and test, used 75/25 split. 
w = Window().partitionBy(lit('a')).orderBy(lit('a'))
df_final = vectors.withColumn("row_num", row_number().over(w)/df.count())
train = df_final.where("row_num <= .75").drop("row_num").select(['features',target])
test = df_final.where("row_num > .75").drop("row_num").select(['features',target]).filter(df_final.canada_lead. isNotNull())


display(train)

# COMMAND ----------

# MAGIC %md #Model Development and Hyperparamter Tuning: Lin Reg 14 Day Window

# COMMAND ----------

# Modelling and Hyperparamter tuning. Below a lienar regression model is initialized, trained, and tuned using a variety of different hyperparameters. 
 
#Initializing linear regression model. The feature column is "features", which was created through vectorization of all of the non target features. labelCol is the target feature, which in this case is the lag of Canada cases by the specified window.
lr = LinearRegression(featuresCol = 'features', labelCol=target)
  
#hyperparameter tuning: creating a list of hyperparameters for the grid search
#---------------------------------------------------------------------------------
#maxIterations refers to the number of iterations used in order to train the algorithm

# elastic net parameters refers to the penalty used in regularization. L2 penalty is used for a value of 0, L1 penalty is used for a value of 1. A combination of L1 and L2 is used for a value of 0.5. These were tested initially, however L2 and none are the only penalty availble when using the Huber Loss function.

#Huber loss is a option for loss function that makes the regression more robust, in the sense that it is much less sensitive to outliers than the Squared Error Loss Function. The Squared Error Loss Function is the sum of squared distances between actual and predicted values  .As the residuals will later show. There are outliers in this data which makes this loss function optimal over squared error loss. Could not run param grid with both as Huber does not allow for elasticNetParam but tested both and Huber was much better.

# Epsilon can only be used with the Huber loss function and is the shape parameter to control the amount of robustness. Through trial and error 4.1 was found as optimal.

# the regularization paramter used in linear regression. A tuning parameter used to control the impact on bias and variance.
#----------------------------------------------------------------------------------------------
param_grid = (ParamGridBuilder() \
               .addGrid(lr.maxIter, [ 5,10,25]) \
               .addGrid(lr.elasticNetParam, [0.0])   \
               .addGrid(lr.regParam, [0.01, 0.1])   \
               .addGrid(lr.loss,["huber"]) \
               .addGrid(lr.epsilon,[4.1]) \
               .addGrid(lr.fitIntercept, [True,False]) \
               .addGrid(lr.standardization, [True,False]) \
               .build())
           
  # Setting the evaluation parameters for the linear regression model. R^2 is defined as the proportion of the variance in the dependent variable that is predictable from the independent. This means that it represents the squared correlation between the predicted and actual cases. The best model will have the highest R2 , closest to 1.0 .
eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="r2")

  # Run rolling k-fold cross validation, provided by Menaka. Using param_grid and r2 as the evaluation metric. 
  
cv = RollingKFoldCV(estimator=lr, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Training model... returns model with optimal hyperparameters that have the best (highest) r2 value.
cvModel = cv.fit(train)

# storing best paramters to be viewed later
bestModel = cvModel.bestModel
bestParams = bestModel.extractParamMap()


# COMMAND ----------

# MAGIC %md #Residual Plot: Lin Reg 14 Day Window

# COMMAND ----------

# residual plot using built in residual plot options, shows good results but decent amount of outliers. Easier to see when Plot Options are selected as databricks has poor graphing support.
display(bestModel,train)

# COMMAND ----------

# MAGIC %md #Best Hyperparameters: Lin Reg 14 Day Window

# COMMAND ----------

#observing best hyperparameters to get a better understanding of the model. This process was used and repeated for tuning as sometimes discluding parameters had a positive impact on r2 and SMAPE.

'Best Param (regParam): ', bestModel._java_obj.getRegParam(), 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam(), 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter(), 'Best Param (solver): ', bestModel._java_obj.getSolver(), 'Best Param (epsilon): ', bestModel._java_obj.getEpsilon(),  bestModel._java_obj.getStandardization(), 'Best Param (standardization): ', bestModel._java_obj.getFitIntercept(), 'Best Param (fitIntercept): '

# COMMAND ----------

# MAGIC %md #Fitting Model on Testing Data: Lin Reg 14 Day Window

# COMMAND ----------

from pyspark.sql.functions import when
# Fitting to testing data using model with optimal hyperparamters found previously. 
results = cvModel.transform(test)
results_train = cvModel.transform(train)
  
# Initializing r2 as eval metric. Next evaluating predictions using r2 as the evaluation metric. 
reg_eval = RegressionEvaluator(labelCol= target, predictionCol= "prediction")
r2_test = reg_eval.evaluate(results, {reg_eval.metricName: "r2"})
r2_train = reg_eval.evaluate(results_train, {reg_eval.metricName: "r2"})


 #Creating a dummy window to be able to order features by date
w = Window().orderBy(lit('A'))
#Get date column as previous one was deleted before vectorization
results = results \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))
results_train = results_train \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))


# removing all predictions with negative case prediction as this is impossible (cant be less than 0 cases). Used 1 case instead of 0 to not mess with the eval metrics that are susceptable to 0s.
pred = results.withColumn("prediction", when(results["prediction"] < 0, 1).otherwise(results["prediction"]))
pred_train = results_train.withColumn("prediction", when(results_train["prediction"] < 0, 1).otherwise(results_train["prediction"]))

# creating a numpy array of actual canada cases (target) as it is easier to perform arithmatic on. Denoted as A. Same done with predictions denoted as F.
A = np.array(pred.select('canada_actual').collect())
F = np.array(pred.select('prediction').collect())
A_train = np.array(pred_train.select('canada_actual').collect())
F_train = np.array(pred_train.select('prediction').collect())

#printing out R2 value and RMSE value of test data. Comparing r2 for train and test to find if data is overfitting , underfitting, or a good fit.
print('r2 test:' + str(r2_test))
print('r2 train:' + str(r2_train))
print('SMAPE test:')
print(smape(A, F))
print('SMAPE train:')
print(smape(A_train, F_train))
 

# COMMAND ----------

# MAGIC %md #Plotting Actual values vs Predicted values: Lin Reg 14 Day Window

# COMMAND ----------

#Observing graph of predictions vs actual for target "canada_cases". It might be neccesary to click Plot Options if Databricks doesnt save settings.
#testing data
# select line chart with canada_actual and prediction under values and date under key

display(pred)

# COMMAND ----------

#training data
# select line chart with canada_actual and prediction under values and date under key
# from this graph it can be seen that the data is slightly overfitting on the training data. This is further supported by the high R2 number of 0.9 for train and lower number off 0.65 for test. This means that the model has high variance and low bias (overfitting). 
display(pred_train)

# COMMAND ----------

# MAGIC %md #Vectorization and Feature Prep: Lin Reg 21 Day Window

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc
import pyspark.sql.functions as func
from pyspark.sql.functions import percent_rank


# creating a list of column names using the original dataframe columns
#first creating new dataframe to work on
column_info = df
#dropping date column as it will not be needed for now
columns_to_drop = ['Date', ]
column_info = column_info.drop(*columns_to_drop)
#creating a list of column names
column_list=column_info.schema.names
#viewing list of column names
#column_info.schema.names

#creating copy of original dataframe to modify
df_new = df
# changing string datatype to double for every column
for c in df.columns:
    # add condition for the cols to be type cast
    df_new=df_new.withColumn(c, df[c].cast('double'))

# Creating lag for target data from features with specified window being 1,7,14 or 21. Done using lead function. Additionally, changing date datatype from double back to a timestamp.
df_new = df_new \
  .withColumn("Date", to_timestamp("Date")) 
df_lag = df_new.withColumn('canada_lead',
                        func.lead(df_new['canada_cases'],21)
                                 .over(Window.orderBy("Date")))

#First Vectorizing (creating a column with all feature data in one). Next , Splitting the data into training data and testing data using a 75/25 split. 
#using canada_lead as target
target = 'canada_lead'

# Creating vectors of feature columns
vectorAssembler = VectorAssembler(inputCols = column_list, outputCol = 'features')
vectors = vectorAssembler.transform(df_lag)

#split into train and test, used 75/25 split. 
w = Window().partitionBy(lit('a')).orderBy(lit('a'))
df_final = vectors.withColumn("row_num", row_number().over(w)/df.count())
train = df_final.where("row_num <= .75").drop("row_num").select(['features',target])
test = df_final.where("row_num > .75").drop("row_num").select(['features',target]).filter(df_final.canada_lead. isNotNull())


display(train)

# COMMAND ----------

# MAGIC %md #Model Development and Hyperparamter Tuning: Lin Reg 21 Day Window

# COMMAND ----------

# Modelling and Hyperparamter tuning. Below a lienar regression model is initialized, trained, and tuned using a variety of different hyperparameters. 
 
#Initializing linear regression model. The feature column is "features", which was created through vectorization of all of the non target features. labelCol is the target feature, which in this case is the lag of Canada cases by the specified window.
lr = LinearRegression(featuresCol = 'features', labelCol=target)
  
#hyperparameter tuning: creating a list of hyperparameters for the grid search
#---------------------------------------------------------------------------------
#maxIterations refers to the number of iterations used in order to train the algorithm

# elastic net parameters refers to the penalty used in regularization. L2 penalty is used for a value of 0, L1 penalty is used for a value of 1. A combination of L1 and L2 is used for a value of 0.5. These were tested initially, however L2 and none are the only penalty availble when using the Huber Loss function.

#Huber loss is a option for loss function that makes the regression more robust, in the sense that it is much less sensitive to outliers than the Squared Error Loss Function. The Squared Error Loss Function is the sum of squared distances between actual and predicted values  .As the residuals will later show. There are outliers in this data which makes this loss function optimal over squared error loss. Could not run param grid with both as Huber does not allow for elasticNetParam but tested both and Huber was much better.

# Epsilon can only be used with the Huber loss function and is the shape parameter to control the amount of robustness. Through trial and error 4.1 was found as optimal.

# the regularization paramter used in linear regression. A tuning parameter used to control the impact on bias and variance.
#----------------------------------------------------------------------------------------------
param_grid = (ParamGridBuilder() \
               .addGrid(lr.maxIter, [ 5,10,25]) \
               .addGrid(lr.elasticNetParam, [0.0])   \
               .addGrid(lr.regParam, [0.01, 0.1])   \
               .addGrid(lr.loss,["huber"]) \
               .addGrid(lr.epsilon,[4.1]) \
               .addGrid(lr.fitIntercept, [True,False]) \
               .addGrid(lr.standardization, [True,False]) \
               .build())
           
  # Setting the evaluation parameters for the linear regression model. R^2 is defined as the proportion of the variance in the dependent variable that is predictable from the independent. This means that it represents the squared correlation between the predicted and actual cases. The best model will have the highest R2 , closest to 1.0 .
eval_ = RegressionEvaluator(labelCol= target, predictionCol= "prediction", metricName="r2")

  # Run rolling k-fold cross validation, provided by Menaka. Using param_grid and r2 as the evaluation metric. 
  
cv = RollingKFoldCV(estimator=lr, estimatorParamMaps=param_grid, evaluator=eval_, numFolds=2, parallelism=2)  

  # Training model... returns model with optimal hyperparameters that have the best (highest) r2 value.
cvModel = cv.fit(train)

# storing best paramters to be viewed later
bestModel = cvModel.bestModel
bestParams = bestModel.extractParamMap()


# COMMAND ----------

# MAGIC %md #Residual Plot: Lin Reg 21 Day Window

# COMMAND ----------

# residual plot using built in residual plot options, shows good results but decent amount of outliers. Easier to see when Plot Options are selected as databricks has poor graphing support.
display(bestModel,train)

# COMMAND ----------

# MAGIC %md #Best Hyperparameters: Lin Reg 21 Day Window

# COMMAND ----------

#observing best hyperparameters to get a better understanding of the model. This process was used and repeated for tuning as sometimes discluding parameters had a positive impact on r2 and SMAPE.

'Best Param (regParam): ', bestModel._java_obj.getRegParam(), 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam(), 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter(), 'Best Param (solver): ', bestModel._java_obj.getSolver(), 'Best Param (epsilon): ', bestModel._java_obj.getEpsilon(),  bestModel._java_obj.getStandardization(), 'Best Param (standardization): ', bestModel._java_obj.getFitIntercept(), 'Best Param (fitIntercept): '

# COMMAND ----------

# MAGIC %md #Fitting Model on Testing Data: Lin Reg 21 Day Window

# COMMAND ----------

from pyspark.sql.functions import when
# Fitting to testing data using model with optimal hyperparamters found previously. 
results = cvModel.transform(test)
results_train = cvModel.transform(train)
  
# Initializing r2 as eval metric. Next evaluating predictions using r2 as the evaluation metric. 
reg_eval = RegressionEvaluator(labelCol= target, predictionCol= "prediction")
r2_test = reg_eval.evaluate(results, {reg_eval.metricName: "r2"})
r2_train = reg_eval.evaluate(results_train, {reg_eval.metricName: "r2"})


 #Creating a dummy window to be able to order features by date
w = Window().orderBy(lit('A'))
#Get date column as previous one was deleted before vectorization
results = results \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))
results_train = results_train \
  .select(col('prediction'), col(target).alias("canada_actual"), 'features') \
  .withColumn('day_num', row_number().over(w)) \
  .withColumn('Date', expr("date_add('2020-05-14', day_num-1)"))


# removing all predictions with negative case prediction as this is impossible (cant be less than 0 cases). Used 1 case instead of 0 to not mess with the eval metrics that are susceptable to 0s.
pred = results.withColumn("prediction", when(results["prediction"] < 0, 1).otherwise(results["prediction"]))
pred_train = results_train.withColumn("prediction", when(results_train["prediction"] < 0, 1).otherwise(results_train["prediction"]))

# creating a numpy array of actual canada cases (target) as it is easier to perform arithmatic on. Denoted as A. Same done with predictions denoted as F.
A = np.array(pred.select('canada_actual').collect())
F = np.array(pred.select('prediction').collect())
A_train = np.array(pred_train.select('canada_actual').collect())
F_train = np.array(pred_train.select('prediction').collect())

#printing out R2 value and RMSE value of test data. Comparing r2 for train and test to find if data is overfitting , underfitting, or a good fit.
print('r2 test:' + str(r2_test))
print('r2 train:' + str(r2_train))
print('SMAPE test:')
print(smape(A, F))
print('SMAPE train:')
print(smape(A_train, F_train))
 #looking at the values it is clear the model is overfitting on the test data, with low bias and high variance. This is supported by a high r2 score in test of 0.9 and a poor r2 in train of 0.65. The SMAPE is satisfactory as 6.67% for test data.

# COMMAND ----------

# MAGIC %md #Plotting Actual values vs Predicted values: Lin Reg 21 Day Window

# COMMAND ----------

#Observing graph of predictions vs actual for target "canada_cases". It might be neccesary to click Plot Options if Databricks doesnt save settings.
#testing data
# select line chart with canada_actual and prediction under values and date under key

display(pred)

# COMMAND ----------

#training data
# select line chart with canada_actual and prediction under values and date under key
# from this graph it can be seen that the data is slightly overfitting on the training data. This is further supported by the high R2 number of 0.9 for train and lower number off 0.65 for test. This means that the model has high variance and low bias (overfitting). 
display(pred_train)
