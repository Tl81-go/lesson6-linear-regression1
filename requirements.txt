
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고 "
    "선형 회귀를 이용해 선택한 연도의 예상 기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

# -------------------------------
# 데이터 불러오기 및 전처리
# -------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV에서 필요한 열을 찾을 수 없습니다: {missing}"
        )

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025].copy()

    # 연도별 평균기온과 유효한 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해는 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오거나 처리하지 못했습니다.")
    st.code(str(e))
    st.stop()

if len(yearly) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        "데이터 주소와 관측 기록을 확인해 주세요."
    )
    st.stop()

# -------------------------------
# 선형 회귀 계산
# -------------------------------
years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# y = a*x + b
slope, intercept = np.polyfit(years, temps, 1)

# 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_year_count = len(yearly)

# -------------------------------
# 연도 슬라이더와 예상 기온
# -------------------------------
st.subheader("🔮 미래의 기온 확인")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "과거 연평균기온의 선형 추세를 연장한 통계적 추정치입니다. "
    "실제 미래 기온과 다를 수 있습니다."
)

# -------------------------------
# 주요 분석 지표
# -------------------------------
st.subheader("📊 분석 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("상관계수", f"{correlation:.3f}")

with col2:
    st.metric("분석에 사용한 연도 수", f"{data_year_count}년")

with col3:
    st.metric(
        "연간 기온 변화 추세",
        f"{slope:+.4f} °C/년"
    )

st.info(
    f"회귀 직선은 총 {data_year_count}개 연도의 자료로 만들었습니다. "
    f"시작 연도는 {start_year}년, 끝 연도는 {end_year}년입니다."
)

# -------------------------------
# 산점도 + 회귀 직선
# -------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
line_years = np.linspace(start_year, end_year, 200)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도 예측값 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측값",
        marker=dict(size=13, symbol="star"),
        hovertemplate=(
            "선택 연도: %{x:.0f}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 회귀식과 해석
# -------------------------------
st.subheader("🧮 회귀식")

st.latex(
    rf"\hat{{y}} = {slope:.5f}x + ({intercept:.3f})"
)

st.write(
    f"- 기울기: 연도가 1년 증가할 때 회귀선상 기온이 "
    f"약 {slope:+.4f} °C 변하는 추세입니다."
)

st.write(
    f"- 상관계수: {correlation:.3f}. "
    "양수이면 연도와 기온이 함께 증가하는 경향, "
    "음수이면 반대 경향을 뜻합니다."
)

st.write(
    "- 주의: 상관관계만으로 연도 변화가 기온 변화의 "
    "직접적인 원인이라고 결론 내릴 수는 없습니다. "
    "이 모델은 다른 기후 요인을 포함하지 않습니다."
)

# -------------------------------
# 분석에 사용한 데이터 확인
# -------------------------------
with st.expander("분석에 사용한 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    st.dataframe(display_df, use_container_width=True)

st.caption(
    "데이터 출처: 서울 기온 CSV · "
    "2025년까지의 기록 중 유효 관측일 300일 이상인 연도"
)
