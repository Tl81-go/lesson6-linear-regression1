
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)


@st.cache_data(ttl=3600)
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 기온 자료 정리
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균 기온과 관측일수 계산
    yearly = (
        df[df["연도"] <= 2025]
        .groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300]

    return yearly.sort_values("연도").reset_index(drop=True)


st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균 기온을 바탕으로 장기적인 기온 변화 추세를 살펴봅니다."
)

try:
    yearly = load_yearly()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.write("데이터 주소와 CSV 열 이름을 확인해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error(
        f"회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        f"현재 유효 연도 수: {len(yearly)}개"
    )
    st.stop()

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

st.write(f"**분석에 사용한 연도 수:** {len(yearly)}개")
st.write(f"**분석 기간:** {start_year}년~{end_year}년")

# 전체 기간 회귀 분석
# 1908년을 기준으로 연도를 변환
x = yearly["연도"] - 1908
y = yearly["연평균기온"]

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 최근 20년 자료 선택
recent_start = end_year - 19
recent = yearly[yearly["연도"] >= recent_start].copy()

if len(recent) < 2:
    st.error("최근 20년의 회귀 분석에 사용할 자료가 부족합니다.")
    st.stop()

# 최근 20년 회귀 분석
recent_slope, recent_intercept = np.polyfit(
    recent["연도"],
    recent["연평균기온"],
    1
)

# 기울기를 100년당 기온 변화로 변환
full_change = slope * 100
recent_change = recent_slope * 100

st.subheader("📈 100년당 기온 변화 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year}년)",
        value=f"{full_change:+.2f} °C",
        help="전체 기간의 회귀 직선을 기준으로 계산한 100년당 기온 변화"
    )

with col2:
    st.metric(
        label=f"최근 20년 ({recent_start}~{end_year}년)",
        value=f"{recent_change:+.2f} °C",
        help="최근 20년의 추세를 100년 기준으로 환산한 값"
    )

st.caption(
    "양수(+)는 상승, 음수(-)는 하락을 의미합니다. "
    "최근 20년의 값은 실제 100년간 관측한 결과가 아니라 "
    "최근 20년의 추세를 100년 기준으로 환산한 값입니다."
)

st.metric("전체 기간 기온과 연도의 상관계수", f"{correlation:.3f}")

# 예측할 연도 선택
st.subheader("🔮 미래 연평균 기온 예측")

target_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

predicted_temp = slope * (target_year - 1908) + intercept

st.metric(
    label=f"{target_year}년 예측 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "예측값은 과거의 직선 추세가 계속된다고 가정한 추정치이며, "
    "실제 미래 기온을 보장하지 않습니다."
)

# 회귀 직선 데이터
plot_df = yearly.copy()
plot_df["회귀 예측 기온"] = (
    slope * (plot_df["연도"] - 1908) + intercept
)

# 관측값 산점도
fig = px.scatter(
    plot_df,
    x="연도",
    y="연평균기온",
    title="서울 연평균 기온과 회귀 직선",
    labels={
        "연도": "연도",
        "연평균기온": "연평균 기온 (°C)"
    },
    hover_data=["관측일수"]
)

# 회귀 직선 추가
fig.add_scatter(
    x=plot_df["연도"],
    y=plot_df["회귀 예측 기온"],
    mode="lines",
    name="전체 기간 회귀 직선"
)

# 선택한 연도의 예측값 표시
fig.add_scatter(
    x=[target_year],
    y=[predicted_temp],
    mode="markers",
    marker={"size": 13},
    name=f"{target_year}년 예측값"
)

fig.update_layout(
    hovermode="x unified",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (°C)"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📝 그래프에서 알 수 있는 것")
st.write(
    "산점은 각 연도의 실제 연평균 기온을 나타내고, "
    "직선은 전체 기간의 기온 변화 추세를 나타냅니다. "
    "100년당 기온 변화 수치를 이용해 장기적인 상승 또는 하락 경향을 "
    "비교할 수 있습니다."
)
