# main.py — 연평균기온에 직선을 맞추고 예측한다
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.title("기온 예측기")


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL)
    df["연도"] = pd.to_datetime(df["날짜"]).dt.year
    grouped = df.groupby("연도")["평균기온"].agg(["mean", "count"]).reset_index()
    # 2026년 수업은 2025년까지, 유효 관측일 300일 이상인 해를 사용한다.
    valid = (grouped["연도"] <= 2025) & (grouped["count"] >= 300)
    return grouped[valid].rename(columns={"mean": "연평균기온"})


yearly = load_yearly()
a, b = np.polyfit(yearly["연도"], yearly["연평균기온"], 1)   # 기울기, 편향

fig = px.scatter(yearly, x="연도", y="연평균기온", opacity=0.6)
fig.add_scatter(x=yearly["연도"], y=a * yearly["연도"] + b, mode="lines", name="회귀 직선")
st.plotly_chart(fig, width="stretch")
st.caption(f"직선을 만든 해: {len(yearly)}개 ({yearly['연도'].min()}~{yearly['연도'].max()}년)")

st.metric("연도와 연평균기온의 상관계수", f"{yearly['연도'].corr(yearly['연평균기온']):.3f}")
year = st.slider("연도를 고르세요", 1900, 2100, 2045)
st.metric(f"{year}년 예상 연평균기온", f"{a * year + b:.1f}℃")
if year < yearly["연도"].min() or year > yearly["연도"].max():
    st.info("학습 범위 밖의 외삽값입니다. 실제 미래 기온을 보장하지 않습니다.")
