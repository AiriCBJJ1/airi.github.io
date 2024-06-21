# 載入必要模組
import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 自定義函数
import indicator_f_Lo2_short
import indicator_forKBar_short

###### (1) 設定網頁標題和樣式 ######
html_temp = """
<div style="background-color:#3872fb;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
    <h2 style="color:white;text-align:center;">Financial Dashboard </h2>
</div>
"""
stc.html(html_temp)

###### (2) 加載數據 ######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(url):
    df = pd.read_pickle(url)
    return df

df_original = load_data('kbars_2454.TW_2022-01-01_2022-11-18.pkl')

###### (3) 設置日期區間選擇 ######
st.subheader("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')

try:
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
except ValueError:
    st.error("日期格式錯誤，請輸入正確的日期格式，如 2022-01-03")
    st.stop()

df = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)]

###### (4) 數據處理 ######
KBar_dic = df.to_dict()
KBar_dic['open'] = np.array(list(KBar_dic['open'].values()))
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
KBar_dic['time'] = np.array([i.to_pydatetime() for i in list(KBar_dic['time'].values())])
KBar_dic['low'] = np.array(list(KBar_dic['low'].values()))
KBar_dic['high'] = np.array(list(KBar_dic['high'].values()))
KBar_dic['close'] = np.array(list(KBar_dic['close'].values()))
KBar_dic['volume'] = np.array(list(KBar_dic['volume'].values()))
KBar_dic['amount'] = np.array(list(KBar_dic['amount'].values()))

###### (5) 設定 K 棒的時間長度 ######
st.subheader("設定一根 K 棒的時間長度(分鐘)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1440, min_value=1, max_value=1440)

Date = start_date.strftime("%Y-%m-%d")
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

KBar_dic = {
    'time': KBar.TAKBar['time'],
    'product': np.repeat('tsmc', KBar.TAKBar['time'].size),
    'open': KBar.TAKBar['open'],
    'high': KBar.TAKBar['high'],
    'low': KBar.TAKBar['low'],
    'close': KBar.TAKBar['close'],
    'volume': KBar.TAKBar['volume']
}

###### (6) 計算各種技術指標 ######
KBar_df = pd.DataFrame(KBar_dic)

# 移動平均線
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2)

KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

# RSI指標
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

# MACD 指標
st.subheader("設定 MACD 指標參數")
fast_period = st.slider('快速指數平滑移動平均線 (EMA) 周期', 1, 50, 12)
slow_period = st.slider('慢速指數平滑移動平均線 (EMA) 周期', 1, 50, 26)
signal_period = st.slider('信號線 (Signal Line) 周期', 1, 50, 9)

def calculate_macd(df, fast_period, slow_period, signal_period):
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd - signal_line
    return macd, signal_line, macd_histogram

KBar_df['MACD'], KBar_df['Signal_Line'], KBar_df['MACD_Histogram'] = calculate_macd(KBar_df, fast_period, slow_period, signal_period)

# 布林通道
st.subheader("設定布林通道參數")
bollinger_period = st.slider('布林通道周期', 1, 50, 20)
std_multiplier = st.slider('標準差倍數', 1.00, 5.00, 2.00)

def calculate_bollinger_bands(df, period, std_multiplier):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * std_multiplier)
    lower_band = sma - (std * std_multiplier)
    return sma, upper_band, lower_band

KBar_df['SMA'], KBar_df['Upper_Band'], KBar_df['Lower_Band'] = calculate_bollinger_bands(KBar_df, bollinger_period, std_multiplier)

# 成交量移動平均
st.subheader("成交量移動平均周期")
volume_ma_period = st.slider('選擇一個整數', 1, 50, 10)
KBar_df['Volume_MA'] = KBar_df['volume'].rolling(window=volume_ma_period).mean()

###### (7) 畫圖 ######
st.subheader("畫圖")

fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('K線圖', '移動平均線', 'RSI 指標', 'MACD 指標', '布林通道'))

# K線圖
fig.add_trace(go.Candlestick(x=KBar_df['time'],
                             open=KBar_df['open'],
                             high=KBar_df['high'],
                             low=KBar_df['low'],
                             close=KBar_df['close'],
                             name='K線圖'),
              row=1, col=1)

# 移動平均線
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_long'], mode='lines', name='長MA', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_short'], mode='lines', name='短MA', line=dict(color='red')), row=2, col=1)

# RSI 指標
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_long'], mode='lines', name='長RSI', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_short'], mode='lines', name='短RSI', line=dict(color='red')), row=3, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_Middle'], mode='lines', name='50', line=dict(color='black', dash='dash')), row=3, col=1)

# MACD 指標
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red')), row=4, col=1)
fig.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker_color='rgba(0, 0, 255, 0.7)'), row=4, col=1)

# 布林通道
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Upper_Band'], mode='lines', name='Upper Band', line=dict(color='blue')), row=5, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Lower_Band'], mode='lines', name='Lower Band', line=dict(color='red')), row=5, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['close'], mode='lines', name='Close Price', line=dict(color='black')), row=5, col=1)

# 設置圖表佈局和樣式
fig.update_layout(xaxis_rangeslider_visible=False, title='技術指標分析', height=1200)

# 顯示圖表
st.plotly_chart(fig, use_container_width=True)

###### (8) 財務報表分析 ######
st.subheader("財務報表分析")
# 在這裡加載和顯示財務報表數據

###### (9) 股價預測 ######
st.subheader("股價預測")
# 在這裡加載和顯示股價預測模型的結果，例如線性回歸

# 用於股價預測的簡單示例：使用線性回歸模型
lr_model = LinearRegression()
X = np.arange(len(KBar_df)).reshape(-1, 1)
y = KBar_df['close'].values
lr_model.fit(X, y)
future_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=30)
future_dates_formatted = [date.strftime('%Y-%m-%d') for date in future_dates]
future_X = np.arange(len(KBar_df), len(KBar_df) + 30).reshape(-1, 1)
future_preds = lr_model.predict(future_X)

fig_lr = go.Figure()
fig_lr.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['close'], mode='lines', name='Historical Data'))
fig_lr.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Predicted Price', line=dict(color='red')))
fig_lr.update_layout(title='股價預測', xaxis_title='Date', yaxis_title='Close Price')
st.plotly_chart(fig_lr, use_container_width=True)
