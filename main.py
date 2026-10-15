
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형회귀로 선택한 연도의 예상 기온을 계산합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

기준연도 = 2025
최소관측일수 = 300

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "평균기온"]
    for col in 필요한열:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 열이 없습니다.")

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 기준 기간은 2025년까지
    df = df[df["날짜"].dt.year <= 기준연도].copy()
    df["연도"] = df["날짜"].dt.year

    return df


try:
    daily_df = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.info(
        "데이터 주소가 열리는지, CSV의 열 이름과 인코딩이 "
        "올바른지 확인해 주세요."
    )
    st.stop()

if daily_df.empty:
    st.error("분석할 수 있는 기온 데이터가 없습니다.")
    st.stop()

# -----------------------------
# 연도별 평균기온 및 관측일수 계산
# -----------------------------
yearly = (
    daily_df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 미만인 연도 제외
model_df = yearly[
    yearly["관측일수"] >= 최소관측일수
].copy()

model_df = model_df.sort_values("연도").reset_index(drop=True)

if len(model_df) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        "데이터 주소와 관측 기록을 확인해 주세요."
    )
    st.stop()

# -----------------------------
# 선형회귀 모델 학습
# -----------------------------
X = model_df[["연도"]]
y = model_df["연평균기온"]

model = LinearRegression()
model.fit(X, y)

model_df["회귀기온"] = model.predict(X)

상관계수 = model_df["연도"].corr(model_df["연평균기온"])
기울기 = float(model.coef_[0])
절편 = float(model.intercept_)

시작연도 = int(model_df["연도"].min())
끝연도 = int(model_df["연도"].max())
사용연도수 = len(model_df)

# -----------------------------
# 분석 기간 및 통계
# -----------------------------
st.subheader("📊 회귀 분석 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("회귀에 사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric(
    "연도와 기온의 상관계수",
    f"{상관계수:.3f}" if pd.notna(상관계수) else "계산 불가"
)

st.caption(
    f"분석 기준: {기준연도}년까지의 데이터 중 "
    f"관측일수가 {최소관측일수}일 이상인 연도만 사용했습니다."
)

if 기울기 > 0:
    st.info(
        f"분석 기간에 연평균기온은 평균적으로 "
        f"1년에 약 {기울기:.3f}°C 상승하는 추세입니다."
    )
elif 기울기 < 0:
    st.info(
        f"분석 기간에 연평균기온은 평균적으로 "
        f"1년에 약 {abs(기울기):.3f}°C 하락하는 추세입니다."
    )
else:
    st.info("회귀 직선에서 연도에 따른 기온 변화 추세가 거의 없습니다.")

# -----------------------------
# 연도별 산점도 + 회귀 직선
# -----------------------------
st.subheader("📈 연도별 연평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=model_df["연도"],
        y=model_df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=model_df[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}°C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
직선연도 = np.linspace(시작연도, 끝연도, 200)
직선기온 = model.predict(
    pd.DataFrame({"연도": 직선연도})
)

fig.add_trace(
    go.Scatter(
        x=직선연도,
        y=직선기온,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("**이 그래프로 알 수 있는 것**")
st.write(
    "점은 각 연도의 실제 연평균기온이고, 직선은 전체적인 "
    "기온 변화 추세입니다. 상관계수가 1에 가까우면 함께 증가하는 "
    "선형 관계가 강하고, -1에 가까우면 반대로 변하는 관계가 강합니다. "
    "0에 가까우면 선형 관계가 약합니다."
)

# -----------------------------
# 연도 선택 및 예상 기온
# -----------------------------
st.divider()
st.subheader("🔮 미래 연도 기온 예측")

선택연도 = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

예상기온 = float(
    model.predict(pd.DataFrame({"연도": [선택연도]}))[0]
)

st.markdown(f"### {선택연도}년 예상 연평균기온")
st.metric(
    label="선형회귀 예상 기온",
    value=f"{예상기온:.2f} °C"
)

# 선택한 연도를 그래프에서 표시
예측선 = go.Figure()

예측선.add_trace(
    go.Scatter(
        x=model_df["연도"],
        y=model_df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6)
    )
)

예측선.add_trace(
    go.Scatter(
        x=[1900, 2100],
        y=model.predict(
            pd.DataFrame({"연도": [1900, 2100]})
        ),
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

예측선.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[예상기온],
        mode="markers+text",
        name="선택 연도 예상값",
        marker=dict(size=15, symbol="star"),
        text=[f"{예상기온:.2f}°C"],
        textposition="top center",
        hovertemplate=(
            f"{선택연도}년 예상기온: "
            "%{y:.2f}°C<extra></extra>"
        )
    )
)

예측선.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    hovermode="closest"
)

st.plotly_chart(예측선, use_container_width=True)

st.warning(
    "주의: 2025년 이후의 기온은 관측값이 아니라 회귀 직선을 "
    "연장해 계산한 값입니다. 실제 미래 기온은 온실가스 배출, "
    "기후 변화, 자연 변동 등 여러 요인에 따라 달라질 수 있습니다."
)

# -----------------------------
# 분석에 사용한 연도별 데이터 확인
# -----------------------------
with st.expander("회귀 분석에 사용한 연도별 데이터 보기"):
    표시자료 = model_df.copy()
    표시자료["연평균기온"] = 표시자료["연평균기온"].round(2)
    표시자료["회귀기온"] = 표시자료["회귀기온"].round(2)
    st.dataframe(표시자료, use_container_width=True)
