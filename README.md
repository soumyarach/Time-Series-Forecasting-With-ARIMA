# Time-Series-Forecasting-With-ARIMA
# What I Built?
I built a time series forecasting model using the ARIMA (AutoRegressive Integrated Moving Average) algorithm to predict future values in a time series dataset. The model is implemented in Python using libraries like statsmodels, pandas, numpy, and matplotlib. The project includes:
- Loading and visualizing time series data.
- Checking for stationarity using the Augmented Dickey-Fuller test.
- Fitting an ARIMA model to the data.
- Forecasting future values.
- Evaluating the model using Mean Squared Error (MSE).
- Saving the model and forecast results.

# How I Built It?
I built this project by:
1. Importing necessary libraries for data manipulation, modeling, and visualization.
2. Loading sample time series data (or using generated data for demonstration).
3. Checking the stationarity of the time series using the ADF test.
4. Differencing the series if necessary to achieve stationarity.
5. Plotting ACF and PACF to help determine the order of the ARIMA model.
6. Fitting an ARIMA model with chosen parameters.
7. Forecasting future values and plotting the results.
8. Evaluating the model performance on a test set.
9. Saving the model and results for future use.

# Why I Built It?
I built this project to:
- Demonstrate how to implement time series forecasting using ARIMA in Python.
- Provide a template for applying ARIMA modeling to other time series datasets.
- Show how to evaluate the performance of an ARIMA model.
- Enable saving and loading of the model for practical applications like predicting stock prices or other time-dependent data.

# Requirements:-
    Python 3.x
    pandas, numpy, matplotlib, statsmodels, sklearn
# Installation Command:-
    pip install pandas numpy matplotlib statsmodels scikit-learn

