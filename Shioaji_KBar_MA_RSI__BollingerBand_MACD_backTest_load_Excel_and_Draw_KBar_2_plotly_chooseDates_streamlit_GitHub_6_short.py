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

# 加載資料
@st.cache
def load_data(url):
    df = pd.read_csv(url)
    return df

df_original = load_data('kbars_2454.TW_2022-01-01_2022-11-18.csv')

# 轉換日期欄位為 datetime (如果尚未轉換)
df_original['Date'] = pd.to_datetime(df_original['Date'], format='%Y/%m/%d')

# 設置日期區間選擇
st.subheader("選擇開始與結束的日期, 區間:2022-01-01 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-01)', '2022-01-01')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')

try:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
except ValueError:
    st.error("日期格式錯誤，請輸入正確的日期格式，如 2022-01-01")
    st.stop()

# 篩選日期範圍
df = df_original[(df_original['Date'] >= start_date) & (df_original['Date'] <= end_date)]

###### (4) 數據處理 ######
KBar_dic = {
    'time': df['Date'].to_numpy(),
    'product': np.repeat('tsmc', len(df)),
    'open': df['Open'].to_numpy(),
    'high': df['High'].to_numpy(),
    'low': df['Low'].to_numpy(),
    'close': df['Close'].to_numpy(),
    'volume': df['Volume'].to_numpy()
}

###### (5) 設定 K 棒的時間長度 ######
st.subheader("設定一根 K 棒的時間長度(分鐘)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1440, min_value=1, max_value=1440)

Date = start_date.strftime("%Y-%m-%d")
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

for i in range(len(KBar_dic['time'])):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

KBar_dic = {
    'time': KBar.TAKBar['time'],
    'product': np.repeat('tsmc', len(KBar.TAKBar['time'])),
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
bollinger_std = st.slider('標準差倍數', 1.0, 5.0, 2.0)

def calculate_bollinger_bands(df, period, std):
    rolling_mean = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std)
    lower_band = rolling_mean - (rolling_std * std)
    return rolling_mean, upper_band, lower_band

KBar_df['Bollinger_Middle'], KBar_df['Bollinger_Upper'], KBar_df['Bollinger_Lower'] = calculate_bollinger_bands(KBar_df, bollinger_period, bollinger_std)

# 成交量加權平均價格(VWAP)
KBar_df['VWAP'] = (KBar_df['close'] * KBar_df['volume']).cumsum() / KBar_df['volume'].cumsum()

###### (7) 呈現數據 ######
# 設置一個 Figure 物件和子圖
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("K 棒圖", "移動平均線 (MA)", "RSI 指標", "MACD 指標和布林通道"))

# 添加 K 棒圖到子圖 1
fig.add_trace(go.Candlestick(x=KBar_df['time'],
                open=KBar_df['open'],
                high=KBar_df['high'],
                low=KBar_df['low'],
                close=KBar_df['close'],
                name="K 棒"), row=1, col=1)

# 添加移動平均線到子圖 2
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_long'], mode='lines', name='長期 MA', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_short'], mode='lines', name='短期 MA', line=dict(color='orange')), row=2, col=1)

# 添加 RSI 指標到子圖 3
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_long'], mode='lines', name='長期 RSI', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_short'], mode='lines', name='短期 RSI', line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_Middle'], mode='lines', name='RSI 中值', line=dict(color='gray', dash='dash')), row=3, col=1)

# 添加 MACD 指標到子圖 4
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=4, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange')), row=4, col=1)

# 添加布林通道到子圖 4
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_Upper'], mode='lines', name='布林通道上軌', line=dict(color='gray', dash='dot')), row=4, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_Middle'], mode='lines', name='布林通道中線', line=dict(color='black')), row=4, col=1)
fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_Lower'], mode='lines', name='布林通道下軌', line=dict(color='gray', dash='dot')), row=4, col=1)

# 設置 x 軸標籤
fig.update_xaxes(title_text="日期", row=4, col=1)

# 設置 y 軸標籤
fig.update_yaxes(title_text="價格", row=1, col=1)
fig.update_yaxes(title_text="價格", row=2, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)

# 設置圖表標題和大小
fig.update_layout(height=1000, width=1200, title_text="技術指標分析")

# 顯示圖表
st.plotly_chart(fig)

###### (8) 顯示原始資料 ######
st.subheader("原始資料")
st.write(df)

###### (9) 顯示處理後的資料 ######
st.subheader("處理後的資料")
st.write(KBar_df)
