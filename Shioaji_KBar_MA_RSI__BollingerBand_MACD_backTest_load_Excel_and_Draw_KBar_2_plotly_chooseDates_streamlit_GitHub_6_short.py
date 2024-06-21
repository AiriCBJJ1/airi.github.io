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

# 自定义函数
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

df_original = load_data('kbars_2330_2022-01-01-2022-11-18.pkl')

# 刪除不必要的列
df_original = df_original.drop('Unnamed: 0', axis=1)

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

df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

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

# 布林帶
st.subheader("設定布林帶參數")
bollinger_period = st.slider('布林帶周期', 1, 50, 20)
bollinger_std = st.slider('標準差倍數', 1.0, 5.0, 2.0)

def calculate_bollinger_bands(df, period, std):
    ma = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    return ma, upper_band, lower_band

KBar_df['Bollinger_MA'], KBar_df['Bollinger_Upper'], KBar_df['Bollinger_Lower'] = calculate_bollinger_bands(KBar_df, bollinger_period, bollinger_std)

# 成交量移動平均
volume_ma_period = st.slider('成交量移動平均周期', 1, 50, 20)
KBar_df['Volume_MA'] = KBar_df['volume'].rolling(window=volume_ma_period).mean()

###### (7) 畫圖 ######
st.subheader("畫圖")

##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_long'], mode='lines', line=dict(color='blue', width=2), name='MA_long'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_short'], mode='lines', line=dict(color='red', width=2), name='MA_short'), secondary_y=True)

    fig1.layout.yaxis2.showgrid = True
    st.plotly_chart(fig1, use_container_width=True)

##### K線圖, RSI指標
with st.expander("K線圖, RSI指標"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig2.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_long'], mode='lines', line=dict(color='blue', width=2), name='RSI_long'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_short'], mode='lines', line=dict(color='red', width=2), name='RSI_short'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI_Middle'], mode='lines', line=dict(color='gray', width=2, dash='dash'), name='RSI_Middle'), secondary_y=False)

    fig2.layout.yaxis2.showgrid = True
    st.plotly_chart(fig2, use_container_width=True)

##### K線圖, MACD指標
with st.expander("MACD 指標"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig3.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'), secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Signal_Line'], mode='lines', line=dict(color='orange', width=2), name='Signal Line'), secondary_y=False)
    fig3.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='green')), secondary_y=False)

    fig3.layout.yaxis2.showgrid = True
    st.plotly_chart(fig3, use_container_width=True)

##### K線圖, 布林帶
with st.expander("布林帶"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_MA'], mode='lines', line=dict(color='blue', width=2), name='MA'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_Upper'], mode='lines', line=dict(color='red', width=2), name='Upper Band'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Bollinger_Lower'], mode='lines', line=dict(color='green', width=2), name='Lower Band'), secondary_y=True)

    fig4.layout.yaxis2.showgrid = True
    st.plotly_chart(fig4, use_container_width=True)

##### 成交量分析
with st.expander("成交量分析"):
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig5.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='blue')), secondary_y=False)
    fig5.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['Volume_MA'], mode='lines', line=dict(color='red', width=2), name='成交量移動平均'), secondary_y=False)

    fig5.layout.yaxis2.showgrid = True
    st.plotly_chart(fig5, use_container_width=True)

##### 假設你有一個包含財務報表數據的 DataFrame
financial_data = load_data('kbars_2330_2022-01-01-2022-11-18.pkl')

with st.expander("财务报表分析"):
    st.write("财务报表数据", financial_data)

    # 选择一个财务指标进行可视化
    financial_metric = st.selectbox('选择一个财务指标', financial_data.columns)
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=financial_data['time'], y=financial_data[financial_metric], name=financial_metric))

    st.plotly_chart(fig6, use_container_width=True)

##### 股價預測
# 準備訓練數據
X = np.arange(len(KBar_df)).reshape(-1, 1)
y = KBar_df['close'].values

model = LinearRegression()
model.fit(X, y)

# 預測未來價格
future_dates = pd.date_range(start=KBar_df['time'].max(), periods=30, freq='D')
X_future = np.arange(len(KBar_df), len(KBar_df) + len(future_dates)).reshape(-1, 1)
y_future = model.predict(X_future)

with st.expander("股价预测"):
    fig7 = make_subplots(specs=[[{"secondary_y": True}]])
    fig7.add_trace(go.Candlestick(x=KBar_df['time'],
                                  open=KBar_df['open'], high=KBar_df['high'],
                                  low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)
    fig7.add_trace(go.Scatter(x=future_dates, y=y_future, mode='lines', line=dict(color='red', width=2), name='预测价格'), secondary_y=True)

    fig7.layout.yaxis2.showgrid = True
    st.plotly_chart(fig7, use_container_width=True)
