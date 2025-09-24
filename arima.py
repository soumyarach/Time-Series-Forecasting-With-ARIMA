import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Generate or load your time series data
np.random.seed(0)
date_range = pd.date_range('2022-01-01', periods=100)
series = pd.Series(np.random.rand(100) + np.arange(100) * 0.1, index=date_range)

# Plot original time series
plt.figure(figsize=(10,6))
plt.plot(series)
plt.title('Original Time Series')
plt.show()

# Check stationarity
result = adfuller(series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Difference the series if it's not stationary
if result[1] > 0.05:
    series_diff = series.diff().dropna()
else:
    series_diff = series

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(10,6))
plot_acf(series_diff, ax=ax[0])
plot_pacf(series_diff, ax=ax[1])
plt.show()

# Determine the order of ARIMA based on ACF and PACF plots
# For this example, let's assume ARIMA(1,1,1)
model = ARIMA(series, order=(1,1,1))
model_fit = model.fit()

# Print out the statistics of the model
print(model_fit.summary())

# Forecast
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Plot forecast
plt.figure(figsize=(10,6))
plt.plot(series, label='Original')
plt.plot(pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, label='Forecast', marker='o', linestyle='--', color='red')
plt.legend()
plt.title('Time Series Forecast')
plt.show()

# Evaluate the model
train_size = int(len(series) * 0.8)
train, test = series[0:train_size], series[train_size:len(series)]
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, forecast)
print('Test MSE: %.3f' % mse)

# Plot test vs forecast
plt.figure(figsize=(10,6))  
plt.plot(test, label='Test')
plt.plot(forecast, label='Forecast', marker='o', linestyle='--', color='red')
plt.legend()
plt.title('Test vs Forecast')
plt.show()


# Save the model    
import pickle
with open('arima_model.pkl', 'wb') as pkl:
    pickle.dump(model_fit, pkl)
# Load the model
with open('arima_model.pkl', 'rb') as pkl:
    loaded_model = pickle.load(pkl)
# Use the loaded model to make predictions
loaded_forecast = loaded_model.forecast(steps=forecast_steps)
print(loaded_forecast)

# Save the model summary to a text file
with open('arima_model_summary.txt', 'w') as f:
    f.write(model_fit.summary().as_text())  
# Save the forecast to a CSV file
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=series.index[-1], periods=len(forecast)+1, freq='D')[1:], columns=['Forecast'])
forecast_df.to_csv('arima_forecast.csv')