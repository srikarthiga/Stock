import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Forecast", layout="centered")

st.title("📈 Stock Forecasting using Prophet")

uploaded_file = st.file_uploader("Upload a CSV file with 'Date' and 'Close' columns", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns")
    else:
        df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        st.write("Data Preview:")
        st.write(df.tail())

        periods = st.slider("Forecast Days", min_value=30, max_value=365, value=90, step=30)

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        st.subheader("Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        st.subheader("Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
