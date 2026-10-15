
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data(ttl=3600)
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 형식 자동 인식
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df[df["연도"] <= 2025]
        .groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    yearly = yearly[yearly["관측일수"] >= 300]
    return yearly.sort_values("연도").reset_index(drop=True)


yearly = load_yearly()

st.title("서울 기온 예측기")

st.write(f"회귀 분석에 사용할 수 있는 연도: {len(yearly)}개")

if len(yearly) < 2:
    st.error("유효한 연도 데이터가 부족합니다. 데이터 주소와 날짜 형식을 확인해 주세요.")
    st.stop()

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

st.write(f"분석 기간: {start_year}년~{end_year}년")

# 전체 기간 회귀 분석
x = yearly["연도"] - 1908
y = yearly["연평균기온"]
slope, intercept = np.polyfit(x, y, 1)

# 최근 20년 회귀 분석
recent = yearly[yearly["연도"] >= end_year - 19]
recent_slope, recent_intercept = np.polyfit(
    recent["연도"], recent["연평균기온"], 1
)

col1, col2 = st.columns(2)
col1.metric("전체 기간 100년당 기온 변화", f"{slope * 100:+.2f} °C")
col2.metric("최근 20년 100년당 기온 변화", f"{recent_slope * 100:+.2f} °C")

# 예측 연도 선택
target_year = st.slider("예측할 연도", 1900, 2100, 2050)
predicted_temp = slope * (target_year - 1908) + intercept

st.metric(f"{target_year}년 예측 연평균 기온", f"{predicted_temp:.2f} °C")

# 실제 관측값과 회귀 직선
plot_df = yearly.copy()
plot_df["회귀 예측 기온"] = (
    slope * (plot_df["연도"] - 1908) + intercept
)

fig = px.scatter(
    plot_df,
    x="연도",
    y="연평균기온",
    title="서울 연평균 기온과 회귀 분석",
    labels={"연도": "연도", "연평균기온": "연평균 기온 (°C)"}
)

fig.add_scatter(
    x=plot_df["연도"],
    y=plot_df["회귀 예측 기온"],
    mode="lines",
    name="회귀 직선"
)

fig.add_scatter(
    x=[target_year],
    y=[predicted_temp],
    mode="markers",
    marker={"size": 12},
    name="선택한 연도의 예측값"
)

st.plotly_chart(fig, use_container_width=True)
