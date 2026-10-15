
@st.cache_data(ttl=3600)
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 형식 자동 인식
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 날짜 중복 제거 후 연도별 평균과 관측일수 계산
    df = df.drop_duplicates(subset=["날짜"])

    grouped = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    valid = (
        (grouped["연도"] <= 2025)
        & (grouped["관측일수"] >= 300)
    )

    return grouped[valid].sort_values("연도").reset_index(drop=True)


yearly = load_yearly()

st.write(f"분석에 사용할 수 있는 연도 수: {len(yearly)}개")

if not yearly.empty:
    st.write(
        f"분석 기간: {int(yearly['연도'].min())}~"
        f"{int(yearly['연도'].max())}년"
    )
