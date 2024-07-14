# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:23:51 2024

@author: edwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset

df = pd.read_csv("C:/Users/c23116383/OneDrive - Cardiff University/Desktop/Timeseries/coursework_data.csv")

df.head()

# Convert 'Call Date to Datetime

df['Call_Date'] = pd.to_datetime(df['Call_Date'], format= '%d/%m/%Y %H:%M:%S')

# Extract Hours and Date from Call Date

df['Hour'] = df['Call_Date'].dt.hour
df['Date'] = df['Call_Date'].dt.date

# Set 'Call Date' as Index

df.index = pd.to_datetime(df.index)

df.set_index('Call_Date', inplace= True)

# Resample data to daily Frequency

daily_calls = df.resample('D')
daily_calls = daily_calls.size().to_frame(name='count')

# Extract day of the Week

daily_calls_n['Dayofweek'] = daily_calls_n.index.dayofweek

# Plot total calls per day of the week

calls_per_day = daily_calls_n.groupby('Dayofweek')['count'].sum()

plt.figure(figsize=(10,6))
calls_per_day.plot(kind='bar', color='blue')
plt.title('Sum of Ambulance Calls per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Calls')
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Extract weekend infomation

df['Dayofweek'] = df['Call_Date'].dt.dayofweek

df['weekend'] = df['Dayofweek'].isin([5,6])

# Group by Hour

hourly_calls = df.groupby('Hour')

hourly_calls.head()

hourly_calls = hourly_calls.size()

hourly_calls.head()

hourly_mean = hourly_calls.mean()

hourly_mean

#  ambulance runs per hr aganist time of day

plt.figure(figsize= (10, 6))
plt.plot(hourly_calls.index, hourly_calls.values, marker = 'o', linestyle = '-')
plt.title('Ambulance Runs Per Hour')
plt.xlabel('hour of Day')
plt.ylabel('Ambulance Runs')
plt.grid(True)
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# Group daily calls by date

daily_calls = df.groupby('Date')

daily_calls.head()

daily_calls = daily_calls.size()

daily_calls

# ambulance runs per day aganist days

plt.figure(figsize =(30, 6))
plt.plot(daily_calls.index, daily_calls.values, marker = 'o', linestyle = '-')
plt.title('Ambulance Runs Per Day')
plt.xlabel('days')
plt.ylabel('Ambulance Runs')

month_labels = daily_calls.index.strftime('%b %Y')
plt.xticks(daily_calls, labels=month_labels)
plt.tight_layout()

plt.show()

# ambulance runs per week aganist days 

df.set_index('Call_Date', inplace=True)

# Resample to weekly frequency

weekly_calls = df.resample('W')

weekly_calls = weekly_calls.size()

plt.figure(figsize= (15, 6))
weekly_calls.plot()
plt.title('weekly Ambulance Runs')
plt.xlabel('week')
plt.ylabel('Number of Ambulance Runs')
plt.grid(True)
plt.tight_layout()
plt.show()


daily_calls = df.resample('D')
daily_calls = daily_calls.size()

plt.figure(figsize= (30, 6))
daily_calls.plot()
weekly_calls.plot()
plt.title('Daily Ambulance Runs')
plt.xlabel('Days')
plt.ylabel('Number of Ambulance Runs')
plt.grid(True)
plt.tight_layout()
plt.show()

#24hr Moving average

hourly = df.resample('H')
hourly = hourly.size()

hourly_ma = hourly.rolling(window=12).mean()

# Plot 24-hour moving average

plt.figure(figsize= (50, 6))
plt.plot(hourly.index, hourly, label ='Centered 24h moving average' )
plt.title('Centered 24hr Moving Average',)
plt.xlabel('Date')
plt.ylabel('Number of Ambulance Runs')
plt.legend()
plt.tight_layout()
plt.show()

#Seven Day Moving Average

seven_day_ma = daily_calls['count'].rolling(window=7).mean()
seven_day_ma_std = daily_calls['count'].rolling(window=7).std()

plt.figure(figsize = (20,6))
#plt.plot(daily_calls.index, daily_calls, label = 'Daily Ambulance runs', alpha = 0.5)
#plt.plot(seven_day_ma.index, seven_day_ma, color = 'green', label='7 day Moving Average')
plt.plot(seven_day_ma_std.index, seven_day_ma_std, color = 'red', label='7day Moving Average STD')
plt.title(' 7 day Moving Average')
plt.xlabel('Date')
plt.ylabel('Number of Ambulance Runs',)
plt.legend()
plt.tight_layout()
plt.show()

# Three Day Moving Average/ 14 day Moving average / 30 day Moving Average

daily_calls.head()

three_day_ma = daily_calls.rolling(window=3).mean()
fourteen_day_ma = daily_calls.rolling(window=14).mean()
thirty_day_ma = daily_calls.rolling(window=30).mean()

plt.figure(figsize =(20,6))
plt.plot(thirty_day_ma.index, thirty_day_ma, color = 'blue', label = '30 day Moving Average')
plt.plot(fourteen_day_ma.index, fourteen_day_ma, color = 'red', label= '14 day Moving Average')
#plt.plot(seven_day_ma.index, seven_day_ma, color = 'green', label='7 day Moving Average')
#plt.plot(three_day_ma.index, three_day_ma, color = 'purple', label= '3 day Moving Average')
plt.title('14 & 30 day Moving Average')
plt.xlabel('Date')
plt.ylabel('Number of Ambulance Runs',)
plt.legend()
plt.tight_layout()
plt.show()

#
#
#
#
#

# weekday vs weekend

df.head()

df['Call_Date'] =pd.to_datetime(df['Call_Date'], format= '%d/%m/%Y %H:%M:%S')

df['Dayofweek'] = df['Call_Date'].dt.dayofweek

df['weekend'] = df['Dayofweek'].isin([5,6])


weekend_runs = df[df['weekend'] == True]

avg_weekend_runs = weekend_runs['weekend'].value_counts() / 54

avg_weekend_runs

weekday_runs = df[df['weekend'] == False]

avg_weekday_runs = weekday_runs['weekend'].value_counts() / 125

avg_weekday_runs

#
#
#
#
#

# Auto correlation function

data_array = daily_calls['count'].values

mean = np.mean(data_array)

variance = np.var(data_array)

q1 = np.percentile(data_array, 25)
q3 = np.percentile(data_array, 75)

median = np.percentile(data_array, 50)

iqr = q3 - q1

quartile_dev = (q3 - q1) / 2

# auto correlation coefficeint

max_lag =14
autocorrelation_cof = []

for lag in range(1, max_lag + 1):
    lag_data = np.roll(data_array, lag)
    cov = np.mean((data_array - mean) * (lag_data - mean))
    autocorrelation = cov / variance
    autocorrelation_cof.append(autocorrelation)

lags = range(0, 14)
autocorrelation_coeffs = [autocorrelation_cof[i] for i in range(14)]

#correlogram

plt.figure(figsize=(10, 6))
plt.bar(lags, autocorrelation_coeffs, color='blue')
plt.title('Correlogram of Autocorrelation Coefficients')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation Coefficient')
plt.xticks(lags)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.show()


# Trend_ local Regression

x_num = np.arange(len(daily_calls))
y = daily_calls.values

loess_estimated = lowess(y, x_num, frac=0.20)

plt.figure(figsize=(15, 6))
plt.plot(daily_calls.index, loess_estimated[:, 1], color= 'red', label= 'Loess Trend')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Number of Runs')
plt.title('Daily Ambulance Runs with LOESS Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# fitting line 

slope, intercept = np.polyfit(x_num, daily_calls.values, 1)

line = slope * x_num + intercept

plt.figure(figsize=(10, 6))
plt.plot(daily_calls.index, daily_calls, label= 'Daily Ambulance Runs', alpha= 0.5)
plt.plot(daily_calls.index, line, color= 'green', label= 'fitted line', linewidth= 2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Number of Runs')
plt.title('Ambulance Runs with Fitted Loess Straight Line Forecast')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

daily_calls['DayOfWeek'] = daily_calls['Call_Date'].dt.dayofweek

#
#
#
#
#

# Decomposition Seasonal Decomposition of Time Series(STL)

stl = STL(daily_calls_diff['FinalDiff'])
result = stl.fit()

fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex = True)
ax[0].plot(daily_calls.index, daily_calls['count'], label = 'Original')
ax[0].set_ylabel('Ambulance Runs')

ax[1].plot(daily_calls.index, result.trend, label= 'Trend', color ='Green')
ax[1].set_ylabel('Trend')

ax[2].plot(daily_calls.index, result.seasonal, label= 'Seasonal', color ='Blue')
ax[2].set_ylabel('Seasonal')

ax[3].plot(daily_calls.index, daily_calls, result.resid, label= 'Residual', color ='Red')
ax[3].set_ylabel('Residual')

plt.xlabel('Date')
plt.suptitle('STL Decomposition of Diffrenced Ambulance Runs',)
plt.tight_layout()
plt.show()

#
#
#
#
#

# ACT & PACF Plots

plot_acf(daily_calls['FinalDiff'].dropna(), lags=40)
plt.show()

plot_pacf(daily_calls['FinalDiff'].dropna(), lags=40)
plt.show()

#
#
#
#
#

# Augmented Dickey-fuller test for Stationality

adf_results = adfuller(daily_calls_diff['FinalDiff'])

adf_stat = adf_results[0]
p_value = adf_results[1]

print(adf_stat)
print(p_value)
print('Critical Values:')
for key, value in adf_results[4].items():
    print('\t%s: %.3f' % (key, value))

#
#
#
#
#    
    
# Cyclic Behaviour using Singular Spectrum Analysis (SSA)

data_array = daily_calls['count'].values

window_size = 7 #trajectory matrix
n = len(data_array)
traj_matrix = np.array([data_array[i:i+window_size] for i in range(n - window_size + 1)])

#singular value decomposition

U, sigma, Vt = np.linalg.svd(traj_matrix) 

k = min(window_size, n - window_size + 1)
selected_eigenvals = sigma[:k]

ssa_data = np.dot(U[:, :k], np.dot(np.diag(selected_eigenvals), Vt[:k])) #SSA DATA

data_array_trimmed = data_array[:len(ssa_data)]

plt.figure(figsize=(30, 6))
plt.plot(daily_calls.index[:len(ssa_data)], data_array_trimmed, label='Original Data')
plt.plot(daily_calls.index[:len(ssa_data)], ssa_data, label= 'Cyclic Behavior', alpha = 0.5)
plt.title('Identified Cyclic Behavior using Singular Spectrum Analysis')
plt.xlabel('Date')
plt.ylabel('Ambulance Runs')
plt.legend()
plt.show()

# Calculating variance#

variance = daily_calls['count'].var()

plt.figure(figsize=(10,6))
plt.plot(daily_calls.index, daily_calls['count'], label='Ambulance Runs')
plt.axhline(y=variance, color='red', linestyle='--', label='variance')
plt.title('Variance of Ambulance Runs Time Series Data')
plt.xlabel('Date')
plt.ylabel('Ambulance Runs')
plt.legend()
plt.show()

# Feature engineering

daily_calls['DayOfWeek'] = daily_calls.index.dayofweek

daily_calls['Date'] = daily_calls['Call_date'].dt.date

df['Date'] =pd.to_datetime(df['Date'])

df.drop('Day of week', axis=1)

df['Month'] = df.index.month

daily_calls['weekend'] = daily_calls['DayOfWeek'].isin([5,6]).astype(int)

daily_calls = df.resample('D')
daily_calls = daily_calls.size().to_frame(name='count')

daily_calls.to_frame(name='count')

trend = loess_estimated[:, 1]

['Trend'] = trend

daily_calls['Trend'] = trend



# Train set - March to July

train_start = '2019-03-01'
train_end = '2019-07-31'

# Validation - August

val_start = '2019-08-01'
val_end = '2019-09-30'

train_data = daily_calls.loc[train_start:train_end]
val_data = daily_calls.loc[val_start:val_end]

#
#
#
#
#


# Seasonal Naive

## find data points form the previous week
last_week_data = train_data.last('7D')

## eeach day to previous week observed value
day_to_last = last_week_data.groupby(last_week_data.index.dayofweek)['count'].last()

validation_data = val_data.copy()

validation_data['forecast'] = validation_data.index.dayofweek.map(day_to_last)

mae_naive = mean_absolute_error(validation_data['count'], validation_data['forecast'])
mse_naive = mean_squared_error(validation_data['count'], validation_data['forecast'])

plt.figure(figsize=(12, 6))
#plt.plot(train_data.index, train_data['count'], label='Train_data')
plt.plot(validation_data.index, validation_data['count'], label='Validation data')
plt.plot(validation_data.index, validation_data['forecast'],linestyle='--', label ='forecast')
plt.title('Seasonal Naive Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Number of Ambulance Calls')
plt.legend()
plt.show()

#
#
#
#
#

#Simple Liner Regression

slr_train_data = daily_calls.loc[train_start:train_end]
slr_val_data = daily_calls.loc[val_start:val_end]


slr_train_data['days'] = (train_data.index - slr_train_data.index.min()).days
slr_val_data['days'] = (slr_val_data.index - slr_val_data.index.min()).days


slr_train_data.index

slr_model = LinearRegression()
slr_model.fit(slr_train_data[['days']], slr_train_data['count'])

slr_val_data['Forecast'] = slr_model.predict(slr_val_data[['days']])

slr_rmse = np.sqrt(mean_squared_error(slr_val_data['count'], slr_val_data['Forecast']))

plt.figure(figsize=(12, 6))
#plt.plot( slr_train_data.index, slr_train_data['count'], label= 'Train')
plt.plot(slr_val_data.index, slr_val_data['count'], label= 'Validation Data')
plt.plot(slr_val_data.index, slr_val_data['Forecast'], label= 'Forecast')
plt.title('Forecast vs actual')
plt.legend()
plt.show

#
#
#
#
#

# Seasonal and Trend Decompomposition Using Loess (STL) + ETS forecasting

stl_result = STL(train_data['count']).fit()

stl_trend = stl_result.trend
stl_seasonal = stl_result.seasonal[-len(val_data):]
stl_residual = stl_result.resid

## ETS

ets_model = ExponentialSmoothing(stl_trend, seasonal_periods=7, trend ='add', seasonal=None)

ets_result = ets_model.fit()

### forecasting


#trun_seasonal = stl_seasonal[-len(val_data):]

stl_ets_forecast = ets_result.forecast(len(val_data))

stl_seasonal.index = val_data.index

forecast_seasonality = stl_ets_forecast + stl_seasonal

#forecast_seasonality.index = val_data.index
#forecast_seasonality = stl_ets_forecast[:len(trun_seasonal)] 

stl_ets_val =val_data.copy()

stl_ets_val['forecast'] = forecast_seasonality

stlf_ets_forecast_err = val_data - forecast_seasonality

stlf_ets_mae = mean_absolute_error(stl_ets_val['count'], stl_ets_val['forecast'])
stlf_ets_mse = mean_squared_error(stl_ets_val['count'], stl_ets_val['forecast'])

plt.figure(figsize=(15, 6))
plt.plot(val_data.index, val_data, label='Actual')
plt.plot(val_data.index, forecast_seasonality, label ='forecast', linestyle='--', color='green')
plt.title('STL WITH ETS Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Number of Ambulance Runs')
plt.legend()
plt.show()

#
#
#
#
#

# Multiplicative Holts-Winters

# Holt-Winters

holtwin_ad_mul = ExponentialSmoothing(train_data['count'], trend= 'add', seasonal='mul', seasonal_periods=7)
holtwin_mul_mul = ExponentialSmoothing(train_data['count'], trend= 'mul', seasonal='mul', seasonal_periods=7)
model = holtwin_mul_mul.fit()

holt_forecast = model.forecast(len(val_data))

holt_mae = mean_absolute_error(val_data['count'], holt_forecast)
holt_mse = mean_squared_error(val_data['count'], holt_forecast)

plt.figure(figsize=(15,6))
plt.plot(val_data.index, val_data['count'], label= 'Validation Data')
plt.plot(val_data.index, forecast, label='Forecast', linestyle'--', color = 'red')
plt.title('Multiplicative Holt-Winters  Forecast - Validation')
plt.xlabel('Date')
plt.ylabel('Ambulance Runs')
plt.legend()
plt.show()

#
#
#
#
#

# Diffrencing

## first order diffrencing

#daily_calls = daily_calls.drop('2019-04-19','2019-04-20','2019-06-05','2019-06-06')

daily_calls = daily_calls.drop('2019-04-19')
daily_calls = daily_calls.drop('2019-04-20')
daily_calls = daily_calls.drop('2019-06-05')
daily_calls = daily_calls.drop('2019-06-06')

daily_calls['FirstDiff'] = daily_calls['count'].diff(1)

## Seasonal Diffrencing
daily_calls['SecondDiff'] = daily_calls['count'].diff(7)

daily_calls['FinalDiff'] = daily_calls['FirstDiff'] - daily_calls['SecondDiff']

daily_calls['FinalDiff'].fillna(method='backfill', inplace=True)

diff_data_start = '2019-03-08'
diff_data_end = '2019-09-30'

daily_calls_diff = daily_calls[diff_data_start:diff_data_end]

### THIS DDNT WORK

#
#
#
#
#

# Box COX Transformation

daily_calls['Transformed'], lambda_value = boxcox(daily_calls['count'])

daily_calls['Inverse_Transformed'] = inv_boxcox(daily_calls['Transformed'], lambda_value)

#
#
#
#
#

# ARIMA

sarima_model = SARIMAX(train_data['count'], order= (2, 0, 1), seasonal_order=(2, 0, 1, 7))
sarima_result = sarima_model.fit()

sarima_result.plot_diagnostics(figsize=(15,12))
plt.show()

## Forecasting

val_start = val_data.index[0]
val_end = val_data.index[60]

forecast_start = pd.Timestamp('2019-10-01')
forecast_end = forecast_start + pd.Timedelta(days=60)


sarima_val = sarima_result.get_prediction(start=val_start, end=val_end)
sarima_val_summary = sarima_val.summary_frame()

## confidence intervals for the forecast
sarima_conf_intervals = sarima_val_summary[['mean_ci_lower', 'mean_ci_upper']]

sarima_forecast_values = sarima_val_summary['mean']

actual_values = val_data['count']

sarima_mse = ((sarima_forecast_values - actual_values) ** 2).mean()
sarima_mae = mean_absolute_error(val_data['count'], sarima_val_summary['mean'])

# 61 day forecast



plt.figure(figsize=(15,6))
#plt.plot(train_data.index, train_data['count'], label='Train_Data')
plt.plot(val_data.index, val_data['count'], label= 'Validation_Data')
plt.plot(sarima_val_summary['mean'], label='Forecast', color='green')
plt.fill_between(sarima_val_summary.index, sarima_conf_intervals['mean_ci_lower'], sarima_conf_intervals['mean_ci_upper'], color='pink', alpha=0.3)
plt.title('SARIMA Validation Plot')
plt.legend(loc='upper left')
plt.show()

