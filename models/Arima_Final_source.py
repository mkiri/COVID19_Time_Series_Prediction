# Databricks notebook source
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import *
from datetime import datetime
from pyspark.sql.functions import to_date, to_timestamp

import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
import sklearn.metrics

# COMMAND ----------

confirmed_cases = spark.table("g6_confirmed_cases_3_csv")

# COMMAND ----------

confirmed_cases_pandas = confirmed_cases.toPandas()
confirmed_cases_pandas['Date'] = pd.to_datetime(confirmed_cases_pandas['Date'], infer_datetime_format = True)
confirmed_cases_init = confirmed_cases_pandas.set_index('Date').asfreq('d')

# COMMAND ----------

canada_final = confirmed_cases_init[["canada_cases"]]
japan_final = confirmed_cases_init[["japan_cases"]]
italy_final = confirmed_cases_init[["italy_cases"]]
uk_final = confirmed_cases_init[["uk_cases"]]
germany_final = confirmed_cases_init[["germany_cases"]]
france_final = confirmed_cases_init[["france_cases"]]

# COMMAND ----------

# MAGIC %md # Canada

# COMMAND ----------

plot_acf(canada_final)

# COMMAND ----------

plot_pacf(canada_final)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning 1 day horizon
def evaluate_arima_model(train, test, arima_order):
  # make predictions
  predictions = list()
  for i in range(len(test)):
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    pred = model_fit.forecast(steps=1)
    pred_1= pred[0]
    predictions.append(pred_1)
    x = test[i] # observation of the ith record in the test set
    train = np.append(canada_final.iloc[(i+1):87].values, x)
	# calculate out of sample error
  #error = np.mean(np.absolute(predictions - test)/ (np.absolute(predictions)+ np.absolute(test)))*100
  error = sklearn.metrics.r2_score(test, predictions)
  return error


def evaluate_models(train, test, p_values, d_values, q_values):
  #best_score, best_cfg = float("inf"), None
  best_score, best_cfg = 0, None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          #sMAPE = evaluate_arima_model(train, test, order)
          r2 = evaluate_arima_model(train, test, order)
          #if sMAPE < best_score:
          if r2 > best_score:
            #best_score, best_cfg = sMAPE, order
            best_score, best_cfg = r2, order
          print('ARIMA%s r2=%.3f' % (order,r2))
        except:
          continue
  print('Best ARIMA%s r2=%.3f' % (best_cfg, best_score))

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning for 1 day
# evaluate parameters
train = canada_final.iloc[0:87].values
test = canada_final.iloc[87:].values
p_values = range(0, 11)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,Variance
df = spark.createDataFrame(canada_final.iloc[0:87])

df.agg({'canada_cases': 'variance'}).show()


# COMMAND ----------

# DBTITLE 1,1 day horizon
train = (canada_final.iloc[0:87].values)
test = (canada_final.iloc[87:].values)
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(0,2,1))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(canada_final.iloc[(i+1):87].values, x)

# COMMAND ----------

#oneD_predictions = DataFrame(predictions).cumsum()
#test_1day = DataFrame(test).cumsum()
oneD_predictions = predictions
test_1day = test
#smape = np.sum(np.absolute(oneD_predictions - test_1day))/ np.sum((oneD_predictions)+ (test_1day))*100
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ ((np.absolute(oneD_predictions)+ np.absolute(test_1day))))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

import sklearn.metrics

r2 = sklearn.metrics.r2_score(test_1day, oneD_predictions)
r2

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning 7 day horizon
def evaluate_arima_model(train, test, arima_order):
  # make predictions
  predictions = list()
  for i in range(len(test)):
    model = ARIMA(train, order=(arima_order))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=7)
    pred_7= pred[0][6] # prediction for 7th day
    predictions.append(pred_7)
    if i<=5:
      x = canada_final.iloc[81:87].values[i]
    else:
      x = test[i] # observation of the ith record in the test set
    train = np.append(canada_final.iloc[(i+1):81].values, x)
  #error = np.mean(np.absolute(predictions - test)/ (np.absolute(predictions)+ np.absolute(test)))*100
  error = sklearn.metrics.r2_score(test, predictions)
  return error


def evaluate_models(train, test, p_values, d_values, q_values):
  #best_score, best_cfg = float("inf"), None
  best_score, best_cfg = 0, None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          #sMAPE = evaluate_arima_model(train, test, order)
          r2 = evaluate_arima_model(train, test, order)
          #if sMAPE < best_score:
          if r2 > best_score:
            #best_score, best_cfg = sMAPE, order
            best_score, best_cfg = r2, order
          print('ARIMA%s r2=%.3f' % (order,r2))
        except:
          continue
  print('Best ARIMA%s r2=%.3f' % (best_cfg, best_score))

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning for 7 day
# evaluate parameters
train = canada_final.iloc[0:81].values
test = canada_final.iloc[87:].values
p_values = range(0, 11)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,Variance
df = spark.createDataFrame(canada_final.iloc[0:81])

df.agg({'canada_cases': 'variance'}).show()


# COMMAND ----------

# DBTITLE 1,7 day horizon
train = canada_final.iloc[0:81].values
test = canada_final.iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  #model = ARIMA(train, order=(1,0,0))
  model = ARIMA(train, order=(10,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = canada_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(canada_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ ((np.absolute(predictions_7day)+ np.absolute(test_7day))))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

r2 = sklearn.metrics.r2_score(test_7day, predictions_7day)
r2

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning 14 day horizon
def evaluate_arima_model(train, test, arima_order):
  # make predictions
  predictions = list()
  for i in range(len(test)):
    model = ARIMA(train, order=(arima_order))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=14)
    pred_14= pred[0][13] # prediction for 14th day
    predictions.append(pred_14)
    if i<=11:
      x = canada_final.iloc[75:87].values[i]
    else:
      x = test[i] # observation of the ith record in the test set
    train = np.append(canada_final.iloc[(i+1):75].values, x)
  #error = np.mean(np.absolute(predictions - test)/ (np.absolute(predictions)+ np.absolute(test)))*100
  error = sklearn.metrics.r2_score(test, predictions)
  return error


def evaluate_models(train, test, p_values, d_values, q_values):
  #best_score, best_cfg = float("inf"), None
  best_score, best_cfg = 0, None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          #sMAPE = evaluate_arima_model(train, test, order)
          r2 = evaluate_arima_model(train, test, order)
          #if sMAPE < best_score:
          if r2 > best_score:
            #best_score, best_cfg = sMAPE, order
            best_score, best_cfg = r2, order
          print('ARIMA%s r2=%.3f' % (order,r2))
        except:
          continue
  print('Best ARIMA%s r2=%.3f' % (best_cfg, best_score))

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning for 14 day
# evaluate parameters
train = canada_final.iloc[0:75].values
test = canada_final['canada_cases'].iloc[87:].values
p_values = range(0, 11)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,Variance
df = spark.createDataFrame(canada_final.iloc[0:75])

df.agg({'canada_cases': 'variance'}).show()


# COMMAND ----------

# DBTITLE 1,14 day horizon
train = canada_final.iloc[0:75].values
test = canada_final['canada_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(9,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = canada_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(canada_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = predictions
test_14day = test
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

r2 = sklearn.metrics.r2_score(test_14day, predictions_14day)
r2

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning 21 day horizon
def evaluate_arima_model(train, test, arima_order):
  # make predictions
  predictions = list()
  for i in range(len(test)):
    model = ARIMA(train, order=(arima_order))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=21)
    pred_21= pred[0][20] # prediction for 21st day
    predictions.append(pred_21)
    if i<=19:
      x = canada_final.iloc[67:87].values[i]
    else:
      x = test[i] # observation of the ith record in the test set
    train = np.append(canada_final.iloc[(i+1):67].values, x)
  #error = np.mean(np.absolute(predictions - test)/ (np.absolute(predictions)+ np.absolute(test)))*100
  error = sklearn.metrics.r2_score(test, predictions)
  return error


def evaluate_models(train, test, p_values, d_values, q_values):
  #best_score, best_cfg = float("inf"), None
  best_score, best_cfg = 0, None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          #sMAPE = evaluate_arima_model(train, test, order)
          r2 = evaluate_arima_model(train, test, order)
          #if sMAPE < best_score:
          if r2 > best_score:
            #best_score, best_cfg = sMAPE, order
            best_score, best_cfg = r2, order
          print('ARIMA%s r2=%.3f' % (order,r2))
        except:
          continue
  print('Best ARIMA%s r2=%.3f' % (best_cfg, best_score))

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning for 21 day
# evaluate parameters
train = canada_final.iloc[0:67].values
test = canada_final['canada_cases'].iloc[87:].values
p_values = range(0, 11)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,Variance
df = spark.createDataFrame(canada_final.iloc[0:67])

df.agg({'canada_cases': 'variance'}).show()


# COMMAND ----------

# DBTITLE 1,21 day horizon
train = canada_final.iloc[0:67].values
test = canada_final['canada_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(2,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=21)
  pred_21= pred[0][20] # prediction for 21st day
  predictions.append(pred_21)
  x = test[i] # observation of the ith record in the test set
  if i<=19:
    x = canada_final.iloc[67:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(canada_final.iloc[(i+1):67].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_21day = predictions
test_21day = test
smape = np.mean(np.absolute(predictions_21day - test_21day)/ (np.absolute(predictions_21day)+ np.absolute(test_21day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_21day, predictions_21day))
rmse

# COMMAND ----------

r2 = sklearn.metrics.r2_score(test_21day, predictions_21day)
r2

# COMMAND ----------

plt.plot(predictions_21day, color='red')
plt.plot(test_21day)

# COMMAND ----------

# MAGIC %md # Japan

# COMMAND ----------

plot_acf(japan_final['japan_cases'])

# COMMAND ----------

plot_pacf(japan_final)

# COMMAND ----------

# evaluate parameters
train = japan_final.iloc[0:87].values
test = japan_final['japan_cases'].iloc[87:].values
p_values = [0, 1, 2, 4, 6, 8, 9, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values) #skipping differencing twice since it produces negative values

# COMMAND ----------

# DBTITLE 1,1 day horizon
train = japan_final.iloc[0:87].values
test = japan_final['japan_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(2,1,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(japan_final.iloc[(i+1):87].values, x)

# COMMAND ----------

oneD_predictions = predictions
test_1day = test
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ (np.absolute(oneD_predictions)+ np.absolute(test_1day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,7 day horizon
train = japan_final.iloc[0:81].values
test = japan_final['japan_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  model = ARIMA(train, order=(2,1,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = japan_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(japan_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ (np.absolute(predictions_7day)+ np.absolute(test_7day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,14 day horizon
train = japan_final.iloc[0:75].values
test = japan_final['japan_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(2,1,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = japan_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(japan_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = np.array(predictions[0:17])
test_14day = test[13:30]
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,30 day horizon
train = japan_final.iloc[0:58].values
test = japan_final['japan_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(2,1,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=30)
	pred_30= pred[0][29] # prediction for 14th day
	predictions.append(pred_30)
	x = test[i] # observation of the ith record in the test set
	train = np.append(train, x)

# COMMAND ----------

# startig 05-25-2020
predictions_30day = np.array(predictions[0:1])
test_30day = test[29:30]
smape = np.mean(np.absolute(predictions_30day - test_30day)/ (np.absolute(predictions_30day)+ np.absolute(test_30day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_30day, predictions_30day))
rmse

# COMMAND ----------

# MAGIC %md # Italy

# COMMAND ----------

plot_acf(italy_final['italy_cases'])

# COMMAND ----------

plot_pacf(italy_final)

# COMMAND ----------

# evaluate parameters
train = italy_final.iloc[0:87].values
test = italy_final['italy_cases'].iloc[87:].values
p_values = [0, 1, 2, 4, 6, 8, 9, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,1 day horizon
train = italy_final.iloc[0:87].values
test = italy_final['italy_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(italy_final.iloc[(i+1):87].values, x)

# COMMAND ----------

oneD_predictions = predictions
test_1day = test
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ (np.absolute(oneD_predictions)+ np.absolute(test_1day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,7 day horizon
train = italy_final.iloc[0:81].values
test = italy_final['italy_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  model = ARIMA(train, order=(1,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = italy_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(italy_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ (np.absolute(predictions_7day)+ np.absolute(test_7day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,14 day horizon
train = italy_final.iloc[0:75].values
test = italy_final['italy_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(1,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = italy_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(italy_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = np.array(predictions[0:17])
test_14day = test[13:30]
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,30 day horizon
train = italy_final.iloc[0:58].values
test = italy_final['italy_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=30)
	pred_30= pred[0][29] # prediction for 14th day
	predictions.append(pred_30)
	x = test[i] # observation of the ith record in the test set
	train = np.append(train, x)

# COMMAND ----------

# startig 05-25-2020
predictions_30day = np.array(predictions[0:1])
test_30day = test[29:30]
smape = np.mean(np.absolute(predictions_30day - test_30day)/ (np.absolute(predictions_30day)+ np.absolute(test_30day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_30day, predictions_30day))
rmse

# COMMAND ----------

# MAGIC %md # UK

# COMMAND ----------

uk_final = uk_final.fillna(method='ffill')

# COMMAND ----------

plt.plot(uk_final)

# COMMAND ----------

plot_acf(uk_final)

# COMMAND ----------

plot_pacf(uk_final)

# COMMAND ----------

def evaluate_arima_model(train, test, arima_order):
  # make predictions
  predictions = list()
  for i in range(len(test)):
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    pred = model_fit.forecast(steps=1)
    pred_1= pred[0]
    predictions.append(pred_1)
    x = test[i] # observation of the ith record in the test set
    train = np.append(uk_final.iloc[(i+1):87].values, x)
	# calculate out of sample error
  #error = np.mean(np.absolute(predictions - test)/ (np.absolute(predictions)+ np.absolute(test)))*100
  error = sklearn.metrics.r2_score(test, predictions)
  return error


def evaluate_models(train, test, p_values, d_values, q_values):
  #best_score, best_cfg = float("inf"), None
  best_score, best_cfg = 0, None
  for p in p_values:
    for d in d_values:
      for q in q_values:
        order = (p,d,q)
        try:
          #sMAPE = evaluate_arima_model(train, test, order)
          r2 = evaluate_arima_model(train, test, order)
          #if sMAPE < best_score:
          if r2 > best_score:
            #best_score, best_cfg = sMAPE, order
            best_score, best_cfg = r2, order
          print('ARIMA%s r2=%.3f' % (order,r2))
        except:
          continue
  print('Best ARIMA%s r2=%.3f' % (best_cfg, best_score))

# COMMAND ----------

# evaluate parameters
train = uk_final.iloc[0:87].values
test = uk_final.iloc[87:].values
p_values = range(0, 11)
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,1 day horizon
train = uk_final.iloc[0:87].values
test = uk_final.iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(uk_final.iloc[(i+1):87].values, x)

# COMMAND ----------

oneD_predictions = predictions
test_1day = test
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ (np.absolute(oneD_predictions)+ np.absolute(test_1day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

r2 = sklearn.metrics.r2_score(test_1day, oneD_predictions)
r2

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,7 day horizon
train = uk_final.iloc[0:81].values
test = uk_final['uk_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  model = ARIMA(train, order=(1,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = uk_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(uk_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ (np.absolute(predictions_7day)+ np.absolute(test_7day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,14 day horizon
train = uk_final.iloc[0:75].values
test = uk_final['uk_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(1,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = uk_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(uk_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = np.array(predictions[0:17])
test_14day = test[13:30]
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,30 day horizon
train = uk_final.iloc[0:87].values
test = uk_final['uk_cases'].iloc[58:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=30)
	pred_30= pred[0][29] # prediction for 14th day
	predictions.append(pred_30)
	x = test[i] # observation of the ith record in the test set
	train = np.append(train, x)

# COMMAND ----------

# startig 05-25-2020
predictions_30day = np.array(predictions[0:1])
test_30day = test[29:30]
smape = np.mean(np.absolute(predictions_30day - test_30day)/ (np.absolute(predictions_30day)+ np.absolute(test_30day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_30day, predictions_30day))
rmse

# COMMAND ----------

# MAGIC %md # Germany

# COMMAND ----------

plot_acf(germany_final['germany_cases'])

# COMMAND ----------

plot_pacf(germany_final)

# COMMAND ----------

# evaluate parameters
train = germany_final.iloc[0:87].values
test = germany_final['germany_cases'].iloc[87:].values
p_values = [0, 1, 2, 4, 6, 8, 9, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,1 day horizon
train = germany_final.iloc[0:87].values
test = germany_final['germany_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,1,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(germany_final.iloc[(i+1):87].values, x)

# COMMAND ----------

oneD_predictions = predictions
test_1day = test
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ (np.absolute(oneD_predictions)+ np.absolute(test_1day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,7 day horizon
train = germany_final.iloc[0:81].values
test = germany_final['germany_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  model = ARIMA(train, order=(1,1,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = germany_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(germany_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ (np.absolute(predictions_7day)+ np.absolute(test_7day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,14 day horizon
train = germany_final.iloc[0:75].values
test = germany_final['germany_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(1,1,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = germany_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(germany_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = np.array(predictions[0:17])
test_14day = test[13:30]
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,30 day horizon
train = germany_final.iloc[0:87].values
test = germany_final['germany_cases'].iloc[58:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(1,1,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=30)
	pred_30= pred[0][29] # prediction for 14th day
	predictions.append(pred_30)
	x = test[i] # observation of the ith record in the test set
	train = np.append(train, x)

# COMMAND ----------

# startig 05-25-2020
predictions_30day = np.array(predictions[0:1])
test_30day = test[29:30]
smape = np.mean(np.absolute(predictions_30day - test_30day)/ (np.absolute(predictions_30day)+ np.absolute(test_30day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_30day, predictions_30day))
rmse

# COMMAND ----------

# MAGIC %md # France

# COMMAND ----------

plot_acf(france_final['france_cases'])

# COMMAND ----------

plot_pacf(france_final)

# COMMAND ----------

# evaluate parameters
train = france_final.iloc[0:87].values
test = france_final['france_cases'].iloc[87:].values
p_values = [0, 1, 2, 4, 6, 8, 9, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, test, p_values, d_values, q_values)

# COMMAND ----------

# DBTITLE 1,1 day horizon
train = france_final.iloc[0:87].values
test = france_final['france_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(10,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=1)
	pred_1= pred[0]
	predictions.append(pred_1)
	x = test[i] # observation of the ith record in the test set
	train = np.append(france_final.iloc[(i+1):87].values, x)

# COMMAND ----------

oneD_predictions = predictions
test_1day = test
smape = np.mean(np.absolute(oneD_predictions - test_1day)/ (np.absolute(oneD_predictions)+ np.absolute(test_1day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_1day, oneD_predictions))
rmse

# COMMAND ----------

plt.plot(oneD_predictions, color='red')
plt.plot(test_1day)

# COMMAND ----------

# DBTITLE 1,7 day horizon
train = france_final.iloc[0:81].values
test = france_final['france_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  
  model = ARIMA(train, order=(10,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=7)
  pred_7= pred[0][6] # prediction for 7th day
  predictions.append(pred_7)
  if i<=5:
    x = france_final.iloc[81:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(france_final.iloc[(i+1):81].values, x)
  #print(train)

# COMMAND ----------

# starting from 05-18
predictions_7day = predictions
test_7day = test
smape = np.mean(np.absolute(predictions_7day - test_7day)/ (np.absolute(predictions_7day)+ np.absolute(test_7day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_7day, predictions_7day))
rmse

# COMMAND ----------

plt.plot(predictions_7day, color='red')
plt.plot(test_7day)

# COMMAND ----------

# DBTITLE 1,14 day horizon
train = france_final.iloc[0:75].values
test = france_final['france_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
  model = ARIMA(train, order=(10,0,0))
  model_fit = model.fit()
  pred = model_fit.forecast(steps=14)
  pred_14= pred[0][13] # prediction for 14th day
  predictions.append(pred_14)
  if i<=11:
    x = france_final.iloc[75:87].values[i]
  else:
    x = test[i] # observation of the ith record in the test set
  train = np.append(france_final.iloc[(i+1):75].values, x)

# COMMAND ----------

# startig 05-25-2020
predictions_14day = np.array(predictions[0:17])
test_14day = test[13:30]
smape = np.mean(np.absolute(predictions_14day - test_14day)/ (np.absolute(predictions_14day)+ np.absolute(test_14day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_14day, predictions_14day))
rmse

# COMMAND ----------

plt.plot(predictions_14day, color='red')
plt.plot(test_14day)

# COMMAND ----------

# DBTITLE 1,30 day horizon
train = france_final.iloc[0:58].values
test = france_final['france_cases'].iloc[87:].values
predictions = list()
for i in range(len(test)):
	model = ARIMA(train, order=(10,0,0))
	model_fit = model.fit()
	pred = model_fit.forecast(steps=30)
	pred_30= pred[0][29] # prediction for 30th day
	predictions.append(pred_30)
	x = test[i] # observation of the ith record in the test set
	train = np.append(train, x)

# COMMAND ----------

# startig 05-25-2020
predictions_30day = np.array(predictions[0:1])
test_30day = test[29:30]
smape = np.mean(np.absolute(predictions_30day - test_30day)/ (np.absolute(predictions_30day)+ np.absolute(test_30day)))*100
smape

# COMMAND ----------

rmse = np.sqrt(mean_squared_error(test_30day, predictions_30day))
rmse

# COMMAND ----------


