import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from neuralprophet import NeuralProphet
from neuralprophet import plot_forecast

START = '2010-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title("Stock Prediction App")

asset_type = ('Stock', 'Forex', 'Index', 'Crypocurrency')

stocks = ('AAPL', "GOOG", "MSFT", "GME")
forex = ('USDJPY=X', 'EURUSD=X', 'GBPUSD=X')
index = ('^DJI', '^IXIC', '^GSPC', '^HSI')
crypo_curr = ('BTC-USD', 'ETH-USD', 'BNB-USD')
period = ('Weekly', 'Monthly', 'Yearly')

selected_asset_type = st.selectbox('Select type of asset to be predicted: ', asset_type)

if selected_asset_type == 'Stock':
    selected_asset = st.selectbox("Select stock for prediction", stocks)
elif selected_asset_type == 'Forex':
    selected_asset = st.selectbox("Select forex for prediction", forex)
elif selected_asset_type == 'Index':
    selected_asset = st.selectbox("Select index for prediction", index)
elif selected_asset_type == 'Crypocurrency':
    selected_asset = st.selectbox("Select crypo for prediction", crypo_curr)   

period_of_prediction = st.selectbox("Select period of prediction", period) 

if period_of_prediction == 'Weekly':
    n_periods = st.slider("Duration of prediction:", 1, 7)
    period = n_periods
elif period_of_prediction == 'Monthly':
    n_periods = st.slider("Duration of prediction:", 7, 365)
    period = n_periods
elif period_of_prediction == 'Yearly':
    n_periods = st.slider("Duration of prediction:", 1, 10)
    period = n_periods * 365

uncertainty_interval = st.slider("Uncertainty interval", min_value = 0.0, max_value = 1.0, value = 0.8)

type_of_model = ('Prophet', 'Neural Prophet')
model_selected = st.selectbox("Select the model to be used for forecast", type_of_model)

enter_button = st.button('Generate forecast!')

if enter_button:

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data...")
    data = load_data(selected_asset)
    data_load_state.text("Loading data...done!")

    st.subheader('Raw data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Forecasting:
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={'Date': 'ds', "Close": 'y'})

    #For Prophet:
    if model_selected == 'Prophet':
        model = Prophet(interval_width=uncertainty_interval)
        model.fit(df_train)
        future = model.make_future_dataframe(periods = period)
        forecast = model.predict(future)      

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write('forecast data')
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        st.write('forecast components')
        fig2 = model.plot_components(forecast)
        st.write(fig2)  

    #For Neural Prophet
    elif model_selected == 'Neural Prophet':
        model = NeuralProphet()
        model.fit(df_train, freq='D')
        future = model.make_future_dataframe(df_train, periods = period)
        forecast = model.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write('forecast data')
        fig1 = plot_forecast.plot(forecast)
        st.plotly_chart(fig1)

        st.write('forecast components')
        fig2 = model.plot_components(forecast)
        st.write(fig2)

    #model.fit(df_train, freq='D')
    #future = model.make_future_dataframe(periods = period)
    #forecast = model.predict(future)
    
    #st.subheader('Forecast data')
    #st.write(forecast.tail())

    #st.write('forecast data')
    #fig1 = plot_plotly(model, forecast)
    #st.plotly_chart(fig1)

    #st.write('forecast components')
    #fig2 = model.plot_components(forecast)
    #st.write(fig2)
    