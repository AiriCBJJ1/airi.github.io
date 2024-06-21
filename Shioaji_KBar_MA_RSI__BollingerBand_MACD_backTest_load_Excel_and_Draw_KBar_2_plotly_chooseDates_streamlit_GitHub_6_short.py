# 載入必要模組
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 自定義函數
# 這裡沒有明確指定要使用的自定義函數，如有需要，可以將它們導入或者在需要的地方使用

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
def load_data(file_path):
    df = pd.read_pickle(file_path)
    return df
    
df_original = load_data('kbars_2454.TW_2022-01-01_2022-11-18.pkl')

# 刪除不必要的列
if 'Unnamed: 0' in df_original.columns:
    df_original = df_original.drop('Unnamed: 0', axis=1)

###### (3) 設置日期區間選擇 ######
st.subheader("選擇開始與結束的日期, 區間:2022-01-01 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-01)', '2022-01-01')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')

try:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
except ValueError:
    st.error("日期格式錯誤，請輸入正確的日期格式，如 2022-01-01")
    st.stop()

df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

###### (4) 顯示數據 ######
st.subheader("顯示數據")
st.write(df.head(10))  # 顯示前10行資料作為示例

