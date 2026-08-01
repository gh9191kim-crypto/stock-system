"""
나스닥 하락 대응 레버리지 전환 백테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QQQ 100% 보유 상태에서 QQQ(나스닥100) 직전 고점 대비 하락폭이
커질수록 QLD(2배)·TQQQ(3배)로 단계적·비가역적(래칫)으로 전환하는
전략을 과거 데이터로 검증하는 Streamlit 앱입니다.

기본 전환 규칙 (사이드바에서 조정 가능):
  하락 10%  → 30% 물량 QLD 전환
  하락 15%  → 30% 물량 TQQQ 전환
  하락 20%  → 나머지 40% 물량 TQQQ 전환
  하락 30%  → QLD 보유분도 전량 TQQQ 전환

실행: streamlit run nasdaq_dip_leverage_backtest.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

TICKERS = ["QQQ", "QLD", "TQQQ", "KRW=X"]
ASSETS = ["QQQ", "QLD", "TQQQ"]

# ══════════════════════════════════════════════════════════════════════════════
#  데이터 & 백테스트 엔진 (Streamlit 비의존 — 단독 테스트 가능)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="시세 데이터를 불러오는 중...")
def fetch_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw[TICKERS].ffill().dropna(how="any")
    return raw


def _target_weights(w1: float, w2: float):
    """단계(0~4)별 목표 비중 테이블. w1=QLD 1차 비중, w2=TQQQ 1차 비중(나머지는 자동 계산)."""
    return [
        {"QQQ": 1.0, "QLD": 0.0, "TQQQ": 0.0},
        {"QQQ": 1 - w1, "QLD": w1, "TQQQ": 0.0},
        {"QQQ": 1 - w1 - w2, "QLD": w1, "TQQQ": w2},
        {"QQQ": 0.0, "QLD": w1, "TQQQ": 1 - w1},
        {"QQQ": 0.0, "QLD": 0.0, "TQQQ": 1.0},
    ]


def _rebalance(shares: dict, row: pd.Series, target_w: dict, fee_rate: float):
    value = sum(shares[a] * row[a] for a in ASSETS)
    turnover = sum(abs(value * target_w[a] - shares[a] * row[a]) for a in ASSETS) / 2
    fee = turnover * fee_rate
    value_after_fee = value - fee
    new_shares = {a: (value_after_fee * target_w[a]) / row[a] if target_w[a] > 0 else 0.0 for a in ASSETS}
    return new_shares, fee


def run_backtest(prices: pd.DataFrame, capital_krw: float, th: dict, w: dict,
                  apply_fx: bool, fee_rate: float, benchmark_only: bool = False):
    """
    th: {'th1','th2','th3','th4'} - 소수(0.10 등) 단위 하락률 기준
    w:  {'w1','w2'} - 소수 단위 비중 (w3 = 1-w1-w2 는 자동 계산)
    """
    df = prices.dropna(subset=ASSETS + ["KRW=X"]).copy()
    fx0 = df["KRW=X"].iloc[0]
    fx_series = df["KRW=X"] if apply_fx else pd.Series(fx0, index=df.index)

    capital_usd = capital_krw / fx0
    shares = {"QQQ": capital_usd / df["QQQ"].iloc[0], "QLD": 0.0, "TQQQ": 0.0}
    target_states = _target_weights(w["w1"], w["w2"])

    peak = df["QQQ"].iloc[0]
    state = 0
    records, events = [], []
    total_fee_usd = 0.0

    for dt, row in df.iterrows():
        peak = max(peak, row["QQQ"])
        dd = (peak - row["QQQ"]) / peak

        if not benchmark_only:
            new_state = state
            if dd >= th["th4"]:
                new_state = 4
            elif dd >= th["th3"]:
                new_state = 3
            elif dd >= th["th2"]:
                new_state = 2
            elif dd >= th["th1"]:
                new_state = 1
            if new_state > state:
                shares, fee = _rebalance(shares, row, target_states[new_state], fee_rate)
                total_fee_usd += fee
                v_after = sum(shares[a] * row[a] for a in ASSETS)
                events.append({
                    "날짜": dt.date(), "하락률": dd, "단계": new_state,
                    "QQQ%": target_states[new_state]["QQQ"] * 100,
                    "QLD%": target_states[new_state]["QLD"] * 100,
                    "TQQQ%": target_states[new_state]["TQQQ"] * 100,
                    "평가금액_USD": v_after, "수수료_USD": fee,
                })
                state = new_state

        value_usd = sum(shares[a] * row[a] for a in ASSETS)
        fx = fx_series.loc[dt]
        records.append({
            "date": dt, "qqq_dd": dd, "value_usd": value_usd, "value_krw": value_usd * fx,
            "w_QQQ": shares["QQQ"] * row["QQQ"] / value_usd if value_usd else 0.0,
            "w_QLD": shares["QLD"] * row["QLD"] / value_usd if value_usd else 0.0,
            "w_TQQQ": shares["TQQQ"] * row["TQQQ"] / value_usd if value_usd else 0.0,
            "state": state,
        })

    daily = pd.DataFrame(records).set_index("date")
    ev = pd.DataFrame(events)
    return daily, ev, total_fee_usd


def calc_metrics(value_krw: pd.Series) -> dict:
    ret = value_krw.pct_change().dropna()
    n_years = (value_krw.index[-1] - value_krw.index[0]).days / 365.25
    total_return = value_krw.iloc[-1] / value_krw.iloc[0] - 1
    cagr = (value_krw.iloc[-1] / value_krw.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    running_max = value_krw.cummax()
    mdd = (value_krw / running_max - 1).min()
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol else np.nan
    return dict(total_return=total_return, cagr=cagr, mdd=mdd, vol=vol, sharpe=sharpe)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
.stApp{background:#f5f7fa;}
.main .block-container{padding-top:1rem;}
div[data-testid="stSidebarContent"]{background:#fff;border-right:1px solid #e2e8f0;}
.kcard{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:14px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.kcard .lbl{font-size:10px;color:#94a3b8;text-transform:uppercase;
  letter-spacing:1.2px;margin-bottom:4px;}
.kcard .val{font-size:18px;font-weight:700;color:#1e293b;}
.kcard .sub{font-size:10px;color:#94a3b8;margin-top:2px;}
</style>
"""


def kcard(lbl, val, sub=""):
    sh = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="kcard"><div class="lbl">{lbl}</div><div class="val">{val}</div>{sh}</div>'


def main():
    st.set_page_config(page_title="나스닥 하락 대응 레버리지 전환 백테스트", page_icon="📉",
                        layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("📉 나스닥 하락 대응 레버리지 전환 백테스트")
    st.caption("QQQ 100% 보유 → QQQ(나스닥100) 직전 고점 대비 하락폭에 따라 QLD·TQQQ로 단계적·비가역적 전환")

    with st.sidebar:
        st.header("⚙️ 설정")
        seed_krw = st.number_input("초기 시드 (원)", min_value=1_000_000, value=600_000_000,
                                    step=10_000_000, format="%d")
        c1, c2 = st.columns(2)
        start_date = c1.date_input("시작일", value=date(2019, 1, 1),
                                    min_value=date(2010, 6, 1), max_value=date.today())
        end_date = c2.date_input("종료일", value=date.today(),
                                  min_value=start_date, max_value=date.today())
        apply_fx = st.checkbox("실제 USD/KRW 환율 변동 반영", value=True,
                                help="해제 시 시작일 환율로 고정하여 미국 증시 성과만 측정합니다.")
        fee_pct = st.slider("전환 시 수수료/슬리피지 (%)", 0.0, 1.0, 0.10, 0.05)

        with st.expander("🎯 하락률 전환 기준 (고급)", expanded=False):
            th1 = st.slider("1차: QLD 전환 하락률 (%)", 1, 40, 10)
            th2 = st.slider("2차: TQQQ 전환 하락률 (%)", th1 + 1, 50, 15)
            th3 = st.slider("3차: TQQQ 잔여물량 전환 하락률 (%)", th2 + 1, 60, 20)
            th4 = st.slider("4차: QLD→TQQQ 전량 전환 하락률 (%)", th3 + 1, 70, 30)
            w1 = st.slider("1차 전환 비중 - QLD (%)", 1, 90, 30)
            w2 = st.slider("2차 전환 비중 - TQQQ (%)", 1, 100 - w1, min(30, 100 - w1))
            w3 = 100 - w1 - w2
            st.caption(f"3차(나머지) 전환 비중: **{w3}%** (자동 계산 = 100 − {w1} − {w2})")

        run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)

    if run_btn:
        try:
            prices = fetch_prices(start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
        except Exception as e:
            st.error(f"시세 데이터를 불러오지 못했습니다: {e}")
            return
        if prices.empty or len(prices) < 2:
            st.error("해당 기간의 시세 데이터가 부족합니다. 기간을 조정해주세요.")
            return

        th = {"th1": th1 / 100, "th2": th2 / 100, "th3": th3 / 100, "th4": th4 / 100}
        w = {"w1": w1 / 100, "w2": w2 / 100}
        fee_rate = fee_pct / 100

        daily, events, total_fee = run_backtest(prices, seed_krw, th, w, apply_fx, fee_rate)
        bench, _, _ = run_backtest(prices, seed_krw, th, w, apply_fx, 0.0, benchmark_only=True)

        st.session_state["daily"] = daily
        st.session_state["events"] = events
        st.session_state["bench"] = bench
        st.session_state["total_fee"] = total_fee
        st.session_state["th_levels"] = (th1, th2, th3, th4)

    if "daily" not in st.session_state:
        st.info("왼쪽에서 설정을 확인하고 **백테스트 실행** 버튼을 눌러주세요.")
        return

    daily = st.session_state["daily"]
    events = st.session_state["events"]
    bench = st.session_state["bench"]
    total_fee = st.session_state["total_fee"]
    th1, th2, th3, th4 = st.session_state["th_levels"]

    m_strat = calc_metrics(daily["value_krw"])
    m_bench = calc_metrics(bench["value_krw"])

    cols = st.columns(5)
    cards = [
        ("최종 평가금액", f"{daily['value_krw'].iloc[-1]:,.0f}원", f"QQQ100%: {bench['value_krw'].iloc[-1]:,.0f}원"),
        ("총 수익률", f"{m_strat['total_return']*100:,.1f}%", f"QQQ100%: {m_bench['total_return']*100:,.1f}%"),
        ("CAGR", f"{m_strat['cagr']*100:,.1f}%", f"QQQ100%: {m_bench['cagr']*100:,.1f}%"),
        ("MDD", f"{m_strat['mdd']*100:,.1f}%", f"QQQ100%: {m_bench['mdd']*100:,.1f}%"),
        ("샤프비율", f"{m_strat['sharpe']:,.2f}", f"QQQ100%: {m_bench['sharpe']:,.2f}"),
    ]
    for col, (lbl, val, sub) in zip(cols, cards):
        col.markdown(kcard(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("####")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["value_krw"], name="전략 (단계적 전환)",
                              line=dict(color="#dc2626", width=2)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench["value_krw"], name="QQQ 100% 보유",
                              line=dict(color="#2563eb", width=1.5, dash="dot")))
    fig.update_layout(title="평가금액 추이 (KRW)", height=420, hovermode="x unified", yaxis_tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=daily.index, y=daily["qqq_dd"] * -100, name="QQQ 고점 대비 하락률",
                               fill="tozeroy", line=dict(color="#64748b")))
    for lv, c in [(th1, "#f59e0b"), (th2, "#f97316"), (th3, "#ef4444"), (th4, "#991b1b")]:
        fig2.add_hline(y=-lv, line_dash="dash", line_color=c, annotation_text=f"-{lv}%")
    fig2.update_layout(title="QQQ 고점 대비 하락률 (%)", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily.index, y=daily["w_QQQ"] * 100, name="QQQ",
                               stackgroup="w", line=dict(color="#93c5fd")))
    fig3.add_trace(go.Scatter(x=daily.index, y=daily["w_QLD"] * 100, name="QLD",
                               stackgroup="w", line=dict(color="#fbbf24")))
    fig3.add_trace(go.Scatter(x=daily.index, y=daily["w_TQQQ"] * 100, name="TQQQ",
                               stackgroup="w", line=dict(color="#f87171")))
    fig3.update_layout(title="자산 배분 비중 추이 (%)", height=300)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📋 전환 이벤트 내역")
    if events.empty:
        st.write("설정한 기간 동안 전환 기준을 충족한 이벤트가 없습니다.")
    else:
        ev_disp = events.copy()
        ev_disp["하락률"] = (ev_disp["하락률"] * 100).round(1).astype(str) + "%"
        ev_disp["평가금액_USD"] = ev_disp["평가금액_USD"].round(0)
        ev_disp["수수료_USD"] = ev_disp["수수료_USD"].round(1)
        st.dataframe(ev_disp, use_container_width=True, hide_index=True)
        st.caption(f"누적 전환 수수료/슬리피지: 약 {total_fee:,.0f} USD")

    with st.expander("⚠️ 백테스트 가정 및 유의사항"):
        st.markdown("""
- 하락률은 **QQQ 종가**의 직전 고점 대비 하락폭을 기준으로 계산합니다 (나스닥종합지수·나스닥100 지수가 아님).
- 전환은 **비가역적(래칫)**입니다: 한 번 전환된 비중은 시장이 반등해도 QQQ로 되돌아가지 않습니다.
- QQQ/QLD/TQQQ 가격은 배당 재투자를 반영한 수정종가(auto_adjust)를 사용합니다.
- 세금은 반영되지 않았습니다. 실제 매매 시 양도소득세 등을 고려해야 합니다.
- QLD/TQQQ는 일별 리밸런싱 레버리지 상품으로, 변동성이 큰 구간에서는 변동성 끌림(volatility decay)으로 인해 기초지수 배수 성과와 괴리가 커질 수 있습니다.
- 과거 성과이며 미래 수익을 보장하지 않습니다.
        """)

    st.download_button("일별 백테스트 결과 CSV 다운로드", daily.to_csv().encode("utf-8-sig"),
                        file_name="nasdaq_dip_leverage_backtest.csv", mime="text/csv")


if __name__ == "__main__":
    main()
