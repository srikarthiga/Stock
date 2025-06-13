import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from prophet import Prophet

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📈 Stock Forecasting using ARIMA, SARIMA, LSTM, and Prophet")

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]
    data.dropna(inplace=True)

    st.subheader("📊 Closing Price Overview")
    st.line_chart(data['Close'])

    # ============================ ARIMA ====================================
    st.header("🔁 ARIMA Forecasting")

    result = adfuller(data['Close'])
    p_value = result[1]

    if p_value > 0.05:
        data_diff = data['Close'].diff().dropna()
    else:
        data_diff = data['Close']

    model_auto = auto_arima(data_diff, seasonal=False, trace=False)
    order = model_auto.order
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.predict(start=len(data)-30, end=len(data)+10)

    fig1, ax1 = plt.subplots()
    ax1.plot(data['Close'], label='Actual')
    ax1.plot(forecast, label='Forecast', color='red')
    ax1.set_title("ARIMA Model Forecast")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # ============================ SARIMA ====================================
    st.header("🔁 SARIMA Forecasting")
    sarima_model = SARIMAX(data['Close'],
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_result = sarima_model.fit()
    sarima_forecast = sarima_result.get_forecast(steps=30)
    conf_int = sarima_forecast.conf_int()

    future_dates = pd.date_range(data.index[-1], periods=31, freq='D')[1:]

    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, data['Close'], label='Observed')
    ax2.plot(future_dates, sarima_forecast.predicted_mean, label='Forecast', color='red')
    ax2.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax2.set_title("SARIMA Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # ============================ LSTM ====================================
    st.header("🔁 LSTM Forecasting")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

    test_data = scaled_data[train_size - 60:]
    x_test = []
    y_test = data['Close'].values[train_size:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    fig3, ax3 = plt.subplots()
    ax3.plot(y_test, label='Actual')
    ax3.plot(predictions, label='Predicted', color='red')
    ax3.set_title("LSTM Forecast")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # ============================ Prophet ====================================
    st.header("🔁 Prophet Forecasting")

    prophet_data = data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(prophet_data)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig4 = m.plot(forecast)
    st.pyplot(fig4)

    fig5 = m.plot_components(forecast)
    st.pyplot(fig5)

else:
    st.warning("📂 Please upload a CSV file with 'Date' and 'Close' columns.")
