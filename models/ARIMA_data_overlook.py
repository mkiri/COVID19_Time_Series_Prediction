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

# COMMAND ----------

# MAGIC %md # Overlook at data

# COMMAND ----------

confirmed_cases_init = spark.table("g6_confirmed_cases_csv")

init_pandas = confirmed_cases_init.toPandas()
init_pandas['Date'] = pd.to_datetime(init_pandas['Date'], infer_datetime_format = True)
init_final = init_pandas.set_index('Date')

train_num = (75/100 * confirmed_cases_init.count())
test_num = confirmed_cases_init.count() - train_num
print(train_num, test_num, confirmed_cases_init.count())

#canada_cases.head(5)

# COMMAND ----------

from pyspark.sql.functions import desc

confirmed_cases = confirmed_cases_init.limit(88)
test = confirmed_cases_init.orderBy(desc("Date")).limit(29).orderBy("Date")

# COMMAND ----------

confirmed_cases_pandas = confirmed_cases.toPandas()
confirmed_cases_pandas['Date'] = pd.to_datetime(confirmed_cases_pandas['Date'], infer_datetime_format = True)
confirmed_cases_final = confirmed_cases_pandas.set_index('Date')

test_pandas = test.toPandas()
test_pandas['Date'] = pd.to_datetime(test_pandas['Date'], infer_datetime_format = True)
test_final = test_pandas.set_index('Date')

test_final.tail(5)

# COMMAND ----------

confirmed_cases_final.tail(5)

# COMMAND ----------

test_final.head(5)

# COMMAND ----------

#canada_cases = confirmed_cases[["Date", "canada_cases"]]
canada_final = confirmed_cases_final[["canada_cases"]]
#japan_cases = confirmed_cases[["Date", "japan_cases"]]
japan_final = confirmed_cases_final[["japan_cases"]]
#italy_cases = confirmed_cases[["Date", "italy_cases"]]
italy_final = confirmed_cases_final[["italy_cases"]]
#uk_cases = confirmed_cases[["Date", "uk_cases"]]
uk_final = confirmed_cases_final[["uk_cases"]]
#germany_cases = confirmed_cases[["Date", "germany_cases"]]
germany_final = confirmed_cases_final[["germany_cases"]]
#france_cases = confirmed_cases[["Date", "france_cases"]]
france_final = confirmed_cases_final[["france_cases"]]

canada_test = test_final[["canada_cases"]]
japan_test = test_final[["japan_cases"]]
italy_test = test_final[["italy_cases"]]
uk_test = test_final[["uk_cases"]]
germany_test = test_final[["germany_cases"]]
france_test = test_final[["france_cases"]]

# COMMAND ----------

fig=canada_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

fig=japan_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

fig=italy_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

fig=uk_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

fig=germany_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

fig=france_final.plot(figsize=(18,4))
display(fig.figure)

# COMMAND ----------

# MAGIC %md # Rolling mean

# COMMAND ----------

# from pyspark.sql.window import Window

# w = Window.orderBy(col('Date').cast('long')).rangeBetween(-6, 0)

# rolmean_can = canada_cases.withColumn('rolling_average', avg("canada_cases").over(w))
# rolmean_can_pandas = rolmean_can.toPandas()
# rolmean_can_pandas['Date'] = pd.to_datetime(rolmean_can_pandas['Date'], infer_datetime_format = True)
# rolmean_can_final = rolmean_can_pandas.set_index('Date')

# rolmean_jap = japan_cases.withColumn('rolling_average', avg("japan_cases").over(w))
# rolmean_jap_pandas = rolmean_jap.toPandas()
# rolmean_jap_pandas['Date'] = pd.to_datetime(rolmean_jap_pandas['Date'], infer_datetime_format = True)
# rolmean_jap_final = rolmean_jap_pandas.set_index('Date')

# rolmean_ita = italy_cases.withColumn('rolling_average', avg("italy_cases").over(w))
# rolmean_ita_pandas = rolmean_ita.toPandas()
# rolmean_ita_pandas['Date'] = pd.to_datetime(rolmean_ita_pandas['Date'], infer_datetime_format = True)
# rolmean_ita_final = rolmean_ita_pandas.set_index('Date')

# rolmean_uk = uk_cases.withColumn('rolling_average', avg("uk_cases").over(w))
# rolmean_uk_pandas = rolmean_uk.toPandas()
# rolmean_uk_pandas['Date'] = pd.to_datetime(rolmean_uk_pandas['Date'], infer_datetime_format = True)
# rolmean_uk_final = rolmean_uk_pandas.set_index('Date')

# rolmean_ger = germany_cases.withColumn('rolling_average', avg("germany_cases").over(w))
# rolmean_ger_pandas = rolmean_ger.toPandas()
# rolmean_ger_pandas['Date'] = pd.to_datetime(rolmean_ger_pandas['Date'], infer_datetime_format = True)
# rolmean_ger_final = rolmean_ger_pandas.set_index('Date')

# rolmean_fra = france_cases.withColumn('rolling_average', avg("france_cases").over(w))
# rolmean_fra_pandas = rolmean_fra.toPandas()
# rolmean_fra_pandas['Date'] = pd.to_datetime(rolmean_fra_pandas['Date'], infer_datetime_format = True)
# rolmean_fra_final = rolmean_fra_pandas.set_index('Date')

# COMMAND ----------

# fig1 = rolmean_can_final.plot(figsize=(18,4)) 
# display(fig1.figure)

# COMMAND ----------

# fig1 = rolmean_jap_final.plot(figsize=(18,4)) 
# display(fig1_jap.figure)

# COMMAND ----------

# fig1 = rolmean_ita_final.plot(figsize=(18,4)) 
# display(fig1_ita.figure)

# COMMAND ----------

# fig1 = rolmean_uk_final.plot(figsize=(18,4)) 
# display(fig1_uk.figure)

# COMMAND ----------

# fig1 = rolmean_ger_final.plot(figsize=(18,4)) 
# display(fig1_ger.figure)

# COMMAND ----------

# fig1 = rolmean_fra_final.plot(figsize=(18,4)) 
# display(fig1_fra.figure)

# COMMAND ----------

# MAGIC %md # Stationarity test

# COMMAND ----------

def test_stationarity(timeseries):
         
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.iloc[:,0].values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print(dftest)
    print(dfoutput)
 
test_stationarity(canada_final)

test_stationarity(japan_final)

test_stationarity(italy_final)

test_stationarity(uk_final)

test_stationarity(germany_final)

test_stationarity(france_final)

# COMMAND ----------

# MAGIC %md # Logs (to increase stationarity)

# COMMAND ----------

#since we can't do logs of 0, change 0s to 1

#canada_final["canada_cases"] = canada_final["canada_cases"].replace(0, 1)

japan_final["japan_cases"] = japan_final["japan_cases"].replace(0, 1)

italy_final["italy_cases"] = italy_final["italy_cases"].replace(0, 1)

uk_final["uk_cases"] = uk_final["uk_cases"].replace(0, 1)

#germany_final["germany_cases"] = germany_final["germany_cases"].replace(0, 1)

france_final["france_cases"] = france_final["france_cases"].replace(0, 1)

# COMMAND ----------

#ts_log = np.log(canada_final)

#fig_log=ts_log.plot(figsize=(18, 6))
#display(fig_log.figure)

# COMMAND ----------

ts_jap_log = np.log(japan_final)

fig_log=ts_jap_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

ts_ita_log = np.log(italy_final)

fig_log=ts_ita_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

ts_uk_log = np.log(uk_final)

fig_log=ts_uk_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

#ts_ger_log = np.log(germany_final)

#fig_log=ts_ger_log.plot(figsize=(18, 6))
#display(fig_log.figure)

# COMMAND ----------

ts_fra_log = np.log(france_final)

fig_log=ts_fra_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

#test_stationarity(ts_log)

test_stationarity(ts_jap_log)

test_stationarity(ts_ita_log)

test_stationarity(ts_uk_log)

#test_stationarity(ts_ger_log)

test_stationarity(ts_fra_log)

# COMMAND ----------

# MAGIC %md # Differencing (to increase stationarity)

# COMMAND ----------

#ts_log['canada_cases'] = ts_log['canada_cases'].diff()
ts_jap_log['japan_cases'] = ts_jap_log['japan_cases'].diff()
ts_ita_log['italy_cases'] = ts_ita_log['italy_cases'].diff()
ts_uk_log['uk_cases'] = ts_uk_log['uk_cases'].diff()
#ts_ger_log['germany_cases'] = ts_ger_log['germany_cases'].diff()
ts_fra_log['france_cases'] = ts_fra_log['france_cases'].diff()

#ts_log['canada_cases'] = ts_log['canada_cases'].fillna(0)
ts_jap_log['japan_cases'] = ts_jap_log['japan_cases'].fillna(0)
ts_ita_log['italy_cases'] = ts_ita_log['italy_cases'].fillna(0)
ts_uk_log['uk_cases'] = ts_uk_log['uk_cases'].diff().fillna(0)
#ts_ger_log['germany_cases'] = ts_ger_log['germany_cases'].fillna(0)
ts_fra_log['france_cases'] = ts_fra_log['france_cases'].fillna(0)

# COMMAND ----------

#fig_log=ts_log.plot(figsize=(18, 6))
#display(fig_log.figure)

# COMMAND ----------

fig_log=ts_jap_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

fig_log=ts_ita_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

fig_log=ts_uk_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

#fig_log=ts_ger_log.plot(figsize=(18, 6))
#display(fig_log.figure)

# COMMAND ----------

fig_log=ts_fra_log.plot(figsize=(18, 6))
display(fig_log.figure)

# COMMAND ----------

#test_stationarity(ts_log)

test_stationarity(ts_jap_log)

test_stationarity(ts_ita_log)

test_stationarity(ts_uk_log)

#test_stationarity(ts_ger_log)

test_stationarity(ts_fra_log)

# COMMAND ----------

# MAGIC %md # Differencing without log

# COMMAND ----------

can_diff = canada_final
#jap_diff = japan_final
#ita_diff = italy_final
#uk_diff = uk_final
#ger_diff = germany_final
#fra_diff = france_final

can_diff['canada_cases'] = can_diff['canada_cases'].diff()
#jap_diff['japan_cases'] = jap_diff['japan_cases'].diff()
#ita_diff['italy_cases'] = ita_diff['italy_cases'].diff()
#uk_diff['uk_cases'] = uk_diff['uk_cases'].diff()
ger_diff['germany_cases'] = ger_diff['germany_cases'].diff()
#fra_diff['france_cases'] = fra_diff['france_cases'].diff()

can_diff['canada_cases'] = can_diff['canada_cases'].fillna(0)
#jap_diff['japan_cases'] = jap_diff['japan_cases'].fillna(0)
#ita_diff['italy_cases'] = ita_diff['italy_cases'].fillna(0)
#uk_diff['uk_cases'] = uk_diff['uk_cases'].diff().fillna(0)
ger_diff['germany_cases'] = ger_diff['germany_cases'].fillna(0)
#fra_diff['france_cases'] = fra_diff['france_cases'].fillna(0)

# COMMAND ----------

ger_diff['germany_cases'] = ger_diff['germany_cases'].diff()
ger_diff['germany_cases'] = ger_diff['germany_cases'].fillna(0)

# COMMAND ----------

test_stationarity(can_diff)

#test_stationarity(jap_diff)

#test_stationarity(ita_diff)

#test_stationarity(uk_diff)

test_stationarity(ger_diff)

#test_stationarity(fra_diff)

# COMMAND ----------

# CHOOSING THE MOST STATIONARY DATA FOR EACH COUNTRY

#country -> original(canada_final), log, log+diff (ts_jap_log), diff (can_diff)

#can -> 0.623984, 0.014365 , 0.545969, [[[1.083107e-22]]]

#jap -> 0.511131, 0.285709, [[[1.256811e-14]]] , 0.019711

#ita -> 0.064367, 0.934665 , [[[0.000011]]] , 0.277255

#uk -> 0.478642, 0.038005, [[[3.460263e-16]]] , 5.414207e-10

#ger -> [[[0.048243]]], 0.302117, 0.720871 , 0.242906

#fra -> 0.507965, 0.076173, [[[3.985235e-25]]] , 0.041409

# COMMAND ----------

can_test_diff = canada_test

can_test_diff['canada_cases'] = can_test_diff['canada_cases'].diff()

can_test_diff['canada_cases'] = can_test_diff['canada_cases'].fillna(0)


# COMMAND ----------

test_stationarity(can_test_diff)

# COMMAND ----------

japan_test["japan_cases"] = japan_test["japan_cases"].replace(0, 1)

italy_test["italy_cases"] = italy_test["italy_cases"].replace(0, 1)

uk_test["uk_cases"] = uk_test["uk_cases"].replace(0, 1)

france_test["france_cases"] = france_test["france_cases"].replace(0, 1)

test_jap_log = np.log(japan_test)

test_ita_log = np.log(italy_test)

test_uk_log = np.log(uk_test)

test_fra_log = np.log(france_test)

test_jap_log['japan_cases'] = test_jap_log['japan_cases'].diff()
test_jap_log['japan_cases'] = test_jap_log['japan_cases'].fillna(0)

test_ita_log['italy_cases'] = test_ita_log['italy_cases'].diff()
test_ita_log['italy_cases'] = test_ita_log['italy_cases'].fillna(0)

test_uk_log['uk_cases'] = test_uk_log['uk_cases'].diff()
test_uk_log['uk_cases'] = test_uk_log['uk_cases'].fillna(0)

test_fra_log['france_cases'] = test_fra_log['france_cases'].diff()
test_fra_log['france_cases'] = test_fra_log['france_cases'].fillna(0)

# COMMAND ----------

test_stationarity(test_jap_log)
test_stationarity(test_ita_log)
test_stationarity(test_uk_log)
test_stationarity(test_fra_log)

test_stationarity(germany_test)

# COMMAND ----------

# MAGIC %md # Final Graphs

# COMMAND ----------

can_diff.drop(can_diff.index[0], inplace=True)

fig=can_diff.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

can_test_diff.drop(can_test_diff.index[0], inplace=True)

fig=can_test_diff.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

ts_jap_log.drop(ts_jap_log.index[0], inplace=True)

fig=ts_jap_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

test_jap_log.drop(test_jap_log.index[0], inplace=True)

fig=test_jap_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

ts_ita_log.drop(ts_ita_log.index[0], inplace=True)

fig=ts_ita_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

test_ita_log.drop(test_ita_log.index[0], inplace=True)

fig=test_ita_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

ts_uk_log.drop(ts_uk_log.index[0], inplace=True)

fig=ts_uk_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

test_uk_log.drop(test_uk_log.index[0], inplace=True)

fig=test_uk_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

germany_final.drop(germany_final.index[0], inplace=True)

fig=germany_final.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

germany_test.drop(germany_test.index[0], inplace=True)

fig=germany_test.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

ts_fra_log.drop(ts_fra_log.index[0], inplace=True)

fig=ts_fra_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

test_fra_log.drop(test_fra_log.index[0], inplace=True)

fig=test_fra_log.plot(figsize=(18, 6))
display(fig.figure)

# COMMAND ----------

# MAGIC %md # ACF, PACF

# COMMAND ----------

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(can_diff)
pyplot.show()

# COMMAND ----------

lag_acf = plot_acf(can_diff, lags=20)
display(lag_acf)

# COMMAND ----------

lag_pacf = plot_pacf(can_diff, lags=20)
display(lag_pacf)

# COMMAND ----------

autocorrelation_plot(ts_jap_log)
pyplot.show()

# COMMAND ----------

lag_jap_acf = plot_acf(ts_jap_log, lags=20)
display(lag_jap_acf)

# COMMAND ----------

lag_pacf = plot_pacf(ts_jap_log, lags=20)
display(lag_pacf)

# COMMAND ----------

autocorrelation_plot(ts_ita_log)
pyplot.show()

# COMMAND ----------

lag_ita_acf = plot_acf(ts_ita_log, lags=20)
display(lag_ita_acf)

# COMMAND ----------

lag_pacf = plot_pacf(ts_ita_log, lags=20)
display(lag_pacf)

# COMMAND ----------

autocorrelation_plot(ts_uk_log)
pyplot.show()

# COMMAND ----------

lag_uk_acf = plot_acf(ts_uk_log, lags=20)
display(lag_uk_acf)

# COMMAND ----------

lag_pacf = plot_pacf(ts_uk_log, lags=20)
display(lag_pacf)

# COMMAND ----------

autocorrelation_plot(germany_final)
pyplot.show()

# COMMAND ----------

lag_ger_acf = plot_acf(germany_final, lags=20)
display(lag_ger_acf)

# COMMAND ----------

lag_pacf = plot_pacf(germany_final, lags=20)
display(lag_pacf)

# COMMAND ----------

autocorrelation_plot(ts_fra_log)
pyplot.show()

# COMMAND ----------

lag_fra_acf = plot_acf(ts_fra_log, lags=20)
display(lag_fra_acf)

# COMMAND ----------

lag_pacf = plot_pacf(ts_fra_log, lags=20)
display(lag_pacf)

# COMMAND ----------

# MAGIC %md # ARIMA SUMMARY

# COMMAND ----------

import math

model = ARIMA(can_diff.astype(float), order=(0, 1, 1)) 
results_ARIMA = model.fit(maxiter=500)  
print(results_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(can_diff)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = can_diff.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["canada_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

model_jap = ARIMA(ts_jap_log.astype(float), order=(1, 1, 0)) 
results_jap_ARIMA = model_jap.fit(maxiter=500) 
print(results_jap_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(ts_jap_log)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = ts_jap_log.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["japan_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

model_ita = ARIMA(ts_ita_log.astype(float), order=(0, 1, 1)) 
results_ita_ARIMA = model_ita.fit(maxiter=500) 
print(results_ita_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(ts_ita_log)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = ts_ita_log.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["italy_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

model_uk = ARIMA(ts_uk_log.astype(float), order=(0, 1, 1)) 
results_uk_ARIMA = model_uk.fit(maxiter=500) 
print(results_uk_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(ts_uk_log)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = ts_uk_log.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["uk_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

model_ger = ARIMA(germany_final.astype(float), order=(1, 1, 1)) 
results_ger_ARIMA = model_ger.fit(maxiter=500) 
print(results_ger_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(germany_final)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = germany_final.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["germany_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

model_fra = ARIMA(ts_fra_log.astype(float), order=(0, 1, 1)) 
results_fra_ARIMA = model_fra.fit(maxiter=500) 
print(results_fra_ARIMA.summary())

# COMMAND ----------

fitted_values  = results_ARIMA.predict(1,len(ts_fra_log)-1,typ='linear')
fitted_values_frame = fitted_values.to_frame()

x = ts_fra_log.merge(fitted_values_frame, how='outer', left_index=True, right_index=True)

fig= x.plot(y = ["france_cases", 0], figsize=(20,5))
display(fig.figure)

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_jap_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_ita_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_uk_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_ger_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

# plot residual errors
residuals = DataFrame(results_fra_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

# COMMAND ----------

model_test = ARIMA(can_diff.astype(float), order=(0, 1, 1)) 
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86,116,typ='linear') 
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(can_diff, 'blue', label='training data') 
ax.plot(can_test_diff, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

#smape
from sklearn import metrics

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(can_test_diff, pred)))

# COMMAND ----------

model_test = ARIMA(ts_jap_log.astype(float), order=(1, 1, 0)) 
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86,116,typ='linear')  
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(ts_jap_log, 'blue', label='training data') 
ax.plot(test_jap_log, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(test_jap_log, pred)))

# COMMAND ----------

model_test = ARIMA(ts_ita_log.astype(float), order=(0, 1, 1))  
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86,116,typ='linear') 
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(ts_ita_log, 'blue', label='training data') 
ax.plot(test_ita_log, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(test_ita_log, pred)))

# COMMAND ----------

model_test = ARIMA(ts_uk_log.astype(float), order=(0, 1, 1))  
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86,116,typ='linear') 
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(ts_uk_log, 'blue', label='training data') 
ax.plot(test_uk_log, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(test_uk_log, pred)))

# COMMAND ----------

model_test = ARIMA(germany_final.astype(float), order=(1, 1, 1))  
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86,116,typ='linear') 
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(germany_final, 'blue', label='training data') 
ax.plot(germany_test, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(germany_test, pred)))

# COMMAND ----------

model_test = ARIMA(ts_fra_log.astype(float), order=(0, 1, 1))  
results_test_ARIMA = model_test.fit(maxiter=500) 

fitted_values_test  = results_test_ARIMA.predict(86, 116, typ='linear') #(1, len(test_fra_log-1))
fitted_values_test_frame = fitted_values_test.to_frame()

fig, ax = plt.subplots(figsize=(20,5))
ax.set(title='Predictions', xlabel='Date', ylabel='Total Number of Cases')
ax.plot(ts_fra_log, 'blue', label='training data') 
ax.plot(test_fra_log, 'black', label='test data actuals')
ax.plot(fitted_values_test_frame, 'r', label='test data forecast')  # np.exp(predictions_series)
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
display(fig.figure)

# COMMAND ----------

pred = fitted_values_test_frame.iloc[3:]

print(np.sqrt(metrics.mean_squared_error(test_fra_log, pred)))

# COMMAND ----------

#p: The number of lag observations included in the model, also called the lag order.
#d: The number of times that the raw observations are differenced, also called the degree of differencing.
#q: The size of the moving average window, also called the order of moving average.
