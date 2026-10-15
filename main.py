
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

기준연도 = 2025
최소관측일수 = 300

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")


@st.cache_data(ttl=3600)
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 데이터만 사용
    df = df[df["연도"] <= 기준연도]

    # 날짜 중복을 제거해 실제 관측일을 계산
    df = df.drop_duplicates(subset=["날짜"])

    grouped = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    valid = grouped["관측일수"] >= 최소관측일수

    return (
        grouped[valid]
        .sort_values("연도")
        .reset_index(drop=True)
    )


# ==========================================
# 데이터 확인
# ==========================================
try:
    yearly = load_yearly()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()

if len(yearly) < 2:
    st.error(
        "회귀 분석에 사용할 수 있는 연도가 부족합니다. "
        "데이터 주소와 관측 기록을 확인해 주세요."
    )
    st.stop()


# ==========================================
# 전체 기간 회귀 분석
# ==========================================
x_all = yearly["연도"].to_numpy(dtype=float)
y_all = yearly["연평균기온"].to_numpy(dtype=float)

a, b = np.polyfit(x_all, y_all, 1)

# 1년당 기울기를 100년당 변화량으로 변환
전체_100년변화 = a * 100

상관계수 = np.corrcoef(x_all, y_all)[0, 1]

시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())


# ==========================================
# 최근 20년 회귀 분석
# 최근 분석 가능 연도를 포함한 20년 구간
# ==========================================
최근시작연도 = 끝연도 - 19

recent = yearly[
    (yearly["연도"] >= 최근시작연도)
    & (yearly["연도"] <= 끝연도)
].copy()

최근_100년변화 = None
a_recent = None
b_recent = None

if len(recent) >= 2 and recent["연도"].nunique() >= 2:
    x_recent = recent["연도"].to_numpy(dtype=float)
    y_recent = recent["연평균기온"].to_numpy(dtype=float)

    a_recent, b_recent = np.polyfit(
        x_recent, y_recent, 1
    )

    최근_100년변화 = a_recent * 100


# ==========================================
# 100년당 기온 변화량을 크게 비교
# ==========================================
st.header("🌡️ 100년에 기온이 얼마나 변할까?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("전체 기간")
    st.metric(
        label=f"{시작연도}~{끝연도}년",
        value=f"{전체_100년변화:+.2f} °C",
        help="전체 기간 회귀 직선의 기울기를 100배 한 값입니다."
    )
    st.caption(
        f"분석 연도: {len(yearly)}개 "
        f"({시작연도}~{끝연도}년)"
    )

with col2:
    st.subheader("최근 20년")
    if 최근_100년변화 is not None:
        st.metric(
            label=f"{int(recent['연도'].min())}~{int(recent['연도'].max())}년",
            value=f"{최근_100년변화:+.2f} °C",
            help="최근 20년 구간 회귀 직선의 기울기를 100배 한 값입니다."
        )
        st.caption(
            f"실제 사용 연도: {len(recent)}개"
        )
    else:
        st.warning(
            "최근 20년 구간에 회귀 분석을 위한 "
            "유효한 연도가 부족합니다."
        )

st.caption(
    "양수(+)는 상승 추세, 음수(-)는 하락 추세를 뜻합니다. "
    "두 값은 각각의 기간에서 구한 직선 기울기를 100년 기준으로 환산한 것입니다."
)

if 최근_100년변화 is not None:
    차이 = 최근_100년변화 - 전체_100년변화

    if 차이 > 0:
        st.write(
            f"최근 20년의 100년당 변화량이 전체 기간보다 "
            f"{차이:.2f}°C 높습니다."
        )
    elif 차이 < 0:
        st.write(
            f"최근 20년의 100년당 변화량이 전체 기간보다 "
            f"{abs(차이):.2f}°C 낮습니다."
        )
    else:
        st.write("두 기간의 100년당 기온 변화량이 같습니다.")


# ==========================================
# 산점도 + 전체 기간 회귀 직선
# ==========================================
st.header("📈 연도별 연평균기온과 회귀 직선")

fig = px.scatter(
    yearly,
    x="연도",
    y="연평균기온",
    opacity=0.65,
    hover_data=["관측일수"],
    labels={
        "연도": "연도",
        "연평균기온": "연평균기온 (°C)",
        "관측일수": "관측일수"
    },
    title="서울 연평균기온과 전체 기간 회귀 직선"
)

line_x = np.linspace(시작연도, 끝연도, 200)

fig.add_scatter(
    x=line_x,
    y=a * line_x + b,
    mode="lines",
    name="전체 기간 회귀 직선",
    line=dict(width=3)
)

fig.update_layout(
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"직선을 만든 해: {len(yearly)}개 "
    f"({시작연도}~{끝연도}년)"
)

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{상관계수:.3f}"
)

st.write(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "양수이면 대체로 시간이 지날수록 기온이 높아지는 관계이고, "
    "음수이면 반대로 낮아지는 관계입니다."
)


# ==========================================
# 연도별 회귀 직선 비교
# ==========================================
st.header("🔎 두 기간의 회귀 직선 비교")

compare_fig = px.scatter(
    yearly,
    x="연도",
    y="연평균기온",
    opacity=0.5,
    labels={
        "연도": "연도",
        "연평균기온": "연평균기온 (°C)"
    },
    title="전체 기간과 최근 20년의 추세"
)

compare_fig.add_scatter(
    x=line_x,
    y=a * line_x + b,
    mode="lines",
    name="전체 기간",
    line=dict(width=3)
)

if a_recent is not None:
    recent_x = np.linspace(
        int(recent["연도"].min()),
        int(recent["연도"].max()),
        100
    )

    compare_fig.add_scatter(
        x=recent_x,
        y=a_recent * recent_x + b_recent,
        mode="lines",
        name="최근 20년",
        line=dict(width=3, dash="dash")
    )

compare_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest"
)

st.plotly_chart(compare_fig, use_container_width=True)


# ==========================================
# 선택 연도 기온 예측
# ==========================================
st.header("🔮 원하는 연도의 예상 기온")

year = st.slider(
    "예상 기온을 확인할 연도를 고르세요.",
    min_value=1900,
    max_value=2100,
    value=2045,
    step=1
)

예상기온 = a * year + b

st.metric(
    f"{year}년 예상 연평균기온",
    f"{예상기온:.1f} °C"
)

if year < 시작연도 or year > 끝연도:
    st.info(
        "학습 범위 밖의 외삽값입니다. "
        "과거의 직선 추세를 연장한 값이며 실제 미래 기온을 보장하지 않습니다."
    )

# 선택 연도를 예측 그래프에 표시
prediction_fig = px.scatter(
    yearly,
    x="연도",
    y="연평균기온",
    opacity=0.5,
    labels={
        "연도": "연도",
        "연평균기온": "연평균기온 (°C)"
    }
)

prediction_x = np.linspace(1900, 2100, 300)

prediction_fig.add_scatter(
    x=prediction_x,
    y=a * prediction_x + b,
    mode="lines",
    name="회귀 직선",
    line=dict(width=3)
)

prediction_fig.add_scatter(
    x=[year],
    y=[예상기온],
    mode="markers+text",
    name="선택한 연도",
    marker=dict(size=14, symbol="star"),
    text=[f"{예상기온:.1f}°C"],
    textposition="top center"
)

prediction_fig.update_layout(
    xaxis=dict(range=[1900, 2100]),
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)"
)

st.plotly_chart(prediction_fig, use_container_width=True)

st.caption(
    "연평균기온은 해당 연도의 유효한 일별 평균기온을 평균한 값입니다. "
    "2025년 이후는 실제 관측값이 아닌 통계적 추정치입니다."
)

# ==========================================
# 분석 데이터 확인
# ==========================================
with st.expander("분석에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.round({"연평균기온": 2}),
        use_container_width=True,
        hide_index=True
    )
