
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# =========================================
# 기본 설정
# =========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)

기준연도 = 2025
최소관측일수 = 300

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온을 분석하고, 선형회귀를 이용해 "
    "선택한 연도의 예상 기온과 장기적인 기온 변화 추세를 확인합니다."
)

# =========================================
# 데이터 불러오기
# =========================================
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

    # 2025년까지의 데이터만 사용
    df = df[df["날짜"].dt.year <= 기준연도].copy()
    df["연도"] = df["날짜"].dt.year

    # 같은 날짜가 중복되면 하루의 평균기온을 하나로 정리
    df = (
        df.groupby(["날짜", "연도"], as_index=False)
        ["평균기온"].mean()
    )

    return df


try:
    daily_df = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.info(
        "데이터 주소, CSV 열 이름, UTF-8 인코딩을 확인해 주세요."
    )
    st.stop()

if daily_df.empty:
    st.error("분석할 수 있는 기온 데이터가 없습니다.")
    st.stop()

# =========================================
# 연도별 평균기온 계산
# =========================================
yearly = (
    daily_df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("날짜", "nunique")
    )
    .reset_index()
)

# 관측일수가 300일 이상인 연도만 사용
model_df = yearly[
    (yearly["연도"] <= 기준연도)
    & (yearly["관측일수"] >= 최소관측일수)
].copy()

model_df = (
    model_df.sort_values("연도")
    .reset_index(drop=True)
)

if len(model_df) < 2 or model_df["연도"].nunique() < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        "데이터 주소와 관측 기록을 확인해 주세요."
    )
    st.stop()

# =========================================
# 전체 기간 선형회귀
# =========================================
X = model_df[["연도"]]
y = model_df["연평균기온"]

model = LinearRegression()
model.fit(X, y)

model_df["회귀기온"] = model.predict(X)

상관계수 = model_df["연도"].corr(
    model_df["연평균기온"]
)

시작연도 = int(model_df["연도"].min())
끝연도 = int(model_df["연도"].max())
사용연도수 = len(model_df)

# 연간 기울기 × 100 = 100년당 기온 변화
전체_100년변화 = float(model.coef_[0]) * 100

# =========================================
# 최근 20년 선형회귀
# =========================================
# 마지막 분석 가능 연도를 포함한 20년 구간
최근시작연도 = 끝연도 - 19

recent_df = model_df[
    (model_df["연도"] >= 최근시작연도)
    & (model_df["연도"] <= 끝연도)
].copy()

최근모델 = None
최근_100년변화 = None

if (
    len(recent_df) >= 2
    and recent_df["연도"].nunique() >= 2
):
    최근모델 = LinearRegression()
    최근모델.fit(
        recent_df[["연도"]],
        recent_df["연평균기온"]
    )

    최근_100년변화 = float(
        최근모델.coef_[0]
    ) * 100

# =========================================
# 100년당 기온 변화 비교
# =========================================
st.header("🌡️ 100년 동안 기온은 얼마나 변할까?")

왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    st.subheader("전체 분석 기간")
    st.metric(
        label=f"{시작연도}~{끝연도}년 추세",
        value=f"{전체_100년변화:+.2f} °C",
        help="전체 분석 기간의 회귀 직선 기울기를 100년 기준으로 환산한 값"
    )
    st.caption(
        f"사용한 연도: {사용연도수}년 · "
        f"관측일 {최소관측일수}일 이상"
    )

with 오른쪽:
    st.subheader("최근 20년")
    if 최근_100년변화 is not None:
        st.metric(
            label=(
                f"{int(recent_df['연도'].min())}"
                f"~{int(recent_df['연도'].max())}년 추세"
            ),
            value=f"{최근_100년변화:+.2f} °C",
            help="최근 분석 가능 구간의 회귀 직선 기울기를 100년 기준으로 환산한 값"
        )
        st.caption(
            f"사용한 연도: {len(recent_df)}년"
        )
    else:
        st.warning(
            "최근 20년 구간에 회귀 분석을 위한 "
            "유효한 연도가 2개 이상 없습니다."
        )

st.info(
    "양수(+)는 상승 추세, 음수(-)는 하락 추세를 의미합니다. "
    "100년당 변화량은 각 기간의 직선 기울기를 100년으로 환산한 값이며, "
    "실제 100년 동안의 기온 변화나 미래의 확정된 예측값은 아닙니다."
)

# =========================================
# 분석 기간 및 상관계수
# =========================================
st.subheader("📊 회귀 분석 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric(
    "연도와 기온의 상관계수",
    f"{상관계수:.3f}" if pd.notna(상관계수) else "계산 불가"
)

# =========================================
# 전체 기간 산점도 + 회귀 직선
# =========================================
st.header("📈 연도별 연평균기온과 회귀 직선")

fig = go.Figure()

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

직선연도 = np.linspace(시작연도, 끝연도, 200)
직선기온 = model.predict(
    pd.DataFrame({"연도": 직선연도})
)

fig.add_trace(
    go.Scatter(
        x=직선연도,
        y=직선기온,
        mode="lines",
        name="전체 기간 회귀 직선",
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
    "각 점은 실제 연평균기온이고 직선은 전체 기간의 기온 변화 추세입니다. "
    "상관계수가 1에 가까우면 연도와 기온이 함께 증가하는 선형 관계가 강하고, "
    "-1에 가까우면 반대 방향의 선형 관계가 강합니다. "
    "0에 가까우면 선형 관계가 약합니다."
)

# =========================================
# 전체 기간 vs 최근 20년 회귀 직선 비교
# =========================================
st.header("🔎 전체 기간과 최근 20년 추세 비교")

비교fig = go.Figure()

비교fig.add_trace(
    go.Scatter(
        x=model_df["연도"],
        y=model_df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>"
    )
)

비교fig.add_trace(
    go.Scatter(
        x=직선연도,
        y=직선기온,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3)
    )
)

if 최근모델 is not None:
    최근직선연도 = np.linspace(
        int(recent_df["연도"].min()),
        int(recent_df["연도"].max()),
        100
    )

    최근직선기온 = 최근모델.predict(
        pd.DataFrame({"연도": 최근직선연도})
    )

    비교fig.add_trace(
        go.Scatter(
            x=최근직선연도,
            y=최근직선기온,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash")
        )
    )

비교fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(비교fig, use_container_width=True)

if 최근_100년변화 is not None:
    차이 = 최근_100년변화 - 전체_100년변화

    st.write(
        f"최근 20년의 100년당 기온 변화량은 전체 기간보다 "
        f"{abs(차이):.2f}°C "
        f"{'높습니다.' if 차이 > 0 else '낮습니다.' if 차이 < 0 else '차이가 없습니다.'}"
    )

st.caption(
    "최근 20년 분석은 마지막 분석 가능 연도를 포함한 20년 구간 중 "
    "관측일수 300일 이상인 연도만 사용합니다. "
    "따라서 실제 회귀에 포함된 연도 수는 20년보다 적을 수 있습니다."
)

# =========================================
# 연도 슬라이더로 예상 기온 계산
# =========================================
st.divider()
st.header("🔮 원하는 연도의 예상 기온")

선택연도 = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

예상기온 = float(
    model.predict(
        pd.DataFrame({"연도": [선택연도]})
    )[0]
)

st.subheader(f"{선택연도}년 예상 연평균기온")

st.metric(
    label="선형회귀 예상 기온",
    value=f"{예상기온:.2f} °C"
)

예측선 = go.Figure()

예측선.add_trace(
    go.Scatter(
        x=model_df["연도"],
        y=model_df["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>"
    )
)

# 선택 연도가 포함되도록 회귀 직선 표시
예측연도범위 = np.linspace(1900, 2100, 300)
예측기온범위 = model.predict(
    pd.DataFrame({"연도": 예측연도범위})
)

예측선.add_trace(
    go.Scatter(
        x=예측연도범위,
        y=예측기온범위,
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
            f"{선택연도}년 예상 기온: "
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
    "미래 예상 기온은 2025년까지의 관측자료로 만든 직선을 연장한 "
    "통계적 추정치입니다. 실제 미래 기온은 기후 변화와 여러 환경 요인에 "
    "따라 달라질 수 있으므로 장기적인 참고 자료로만 활용하세요."
)

# =========================================
# 연도별 데이터 확인
# =========================================
with st.expander("📋 회귀 분석에 사용한 연도별 데이터 보기"):
    표시자료 = model_df.copy()
    표시자료["연평균기온"] = 표시자료["연평균기온"].round(2)
    표시자료["회귀기온"] = 표시자료["회귀기온"].round(2)

    st.dataframe(
        표시자료,
        use_container_width=True,
        hide_index=True
    )
