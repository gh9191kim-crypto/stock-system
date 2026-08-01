"""
지금 이 순간 QQQ 기준으로 전략A/B 진입 조건이 어떤 상태인지 확인하는 스크립트.

api/compare.py 의 조건식(하락률·RSI·VIX, 매물대 돌파)을 그대로 사용해 "지금 얼마면
들어가는지", "지금 조건이 충족됐는지"를 알려준다. 매도(청산) 조건은 다루지 않는다 —
이미 보유 중이 아니라 신규 진입을 고민하는 상황을 가정한다.

사용법:
    pip install yfinance pandas numpy
    python3 live_signal_check.py
    python3 live_signal_check.py --dd-th 10 --rsi-th 30 --vix-th 20
    python3 live_signal_check.py --rebound-pct 5 --pullback-pct 3 --breakout-buffer 0.5
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import compare as engine  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="QQQ 현재가 기준 전략A/B 진입 신호 체크")
    p.add_argument("--history-start", type=str, default="2019-01-01",
                    help="고점(전고점)을 계산할 기준 시작일")
    p.add_argument("--dd-th", type=float, default=10.0, help="전략A 진입 하락률 기준(%%)")
    p.add_argument("--rsi-th", type=float, default=30.0, help="전략A 진입 RSI 이하 기준")
    p.add_argument("--vix-th", type=float, default=20.0, help="전략A 진입 VIX 이상 기준")
    p.add_argument("--rebound-pct", type=float, default=5.0, help="전략B 반등 확인(저점대비 %%)")
    p.add_argument("--pullback-pct", type=float, default=3.0, help="전략B 매물대 확정 눌림목(%%)")
    p.add_argument("--breakout-buffer", type=float, default=0.5, help="전략B 돌파 확인 버퍼(%%)")
    return p.parse_args()


def replay_breakout_phase(qqq: pd.Series, dd_th: float, rebound_pct: float, pullback_pct: float):
    """가장 최근 전고점(사상 최고가) 시점부터 오늘까지 되짚어서 현재 국면(phase)을 계산한다."""
    running_max = qqq.cummax()
    at_high = qqq >= running_max  # 그 날 자체가 신고가였는지
    high_dates = qqq.index[at_high]
    last_ath_date = high_dates[-1] if len(high_dates) else qqq.index[0]
    peak = qqq.loc[last_ath_date]  # last_ath_date 이후로는 이 값이 곧 그 시점까지의 전고점

    phase, episode_low, swing_high = "IDLE", None, None
    for dt in qqq.index[qqq.index >= last_ath_date]:
        px = qqq.loc[dt]
        if phase == "IDLE":
            dd = (peak - px) / peak
            if dd >= dd_th:
                phase, episode_low = "DECLINE", px
        elif phase == "DECLINE":
            episode_low = min(episode_low, px)
            if px >= episode_low * (1 + rebound_pct):
                phase, swing_high = "REBOUND", px
        elif phase == "REBOUND":
            if px < episode_low:
                episode_low, phase, swing_high = px, "DECLINE", None
            else:
                swing_high = max(swing_high, px)
                if px <= swing_high * (1 - pullback_pct):
                    phase = "CONFIRMED"
        elif phase == "CONFIRMED":
            if px < episode_low:
                episode_low, phase, swing_high = px, "DECLINE", None
    return phase, episode_low, swing_high, last_ath_date


def main():
    args = parse_args()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    end_exclusive = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"시세 데이터 조회 중... ({args.history_start} ~ {today})")
    try:
        df = engine.fetch_prices(args.history_start, end_exclusive)
    except Exception as e:
        print(f"\n[오류] 시세 데이터를 불러오지 못했습니다: {e}")
        sys.exit(1)

    if df.empty:
        print("\n[오류] 데이터를 가져오지 못했습니다.")
        sys.exit(1)

    last = df.iloc[-1]
    last_date = df.index[-1].date().isoformat()
    peak = df["QQQ"].max()
    dd = (peak - last["QQQ"]) / peak

    print("\n" + "=" * 60)
    print(f"기준일: {last_date} (조회된 데이터의 가장 최근 거래일)")
    print("=" * 60)
    print(f"QQQ 현재가        : ${last['QQQ']:.2f}")
    print(f"QQQ 전고점({args.history_start}~) : ${peak:.2f}")
    print(f"고점대비 하락률    : {dd*100:.2f}%")
    print(f"RSI(14)           : {last['RSI']:.1f}")
    print(f"VIX               : {last['VIX']:.1f}")
    print(f"TQQQ 현재가        : ${last['TQQQ']:.2f}")
    if pd.notna(last["BULZ"]):
        print(f"BULZ 현재가        : ${last['BULZ']:.2f}")
    else:
        print("BULZ 현재가        : 데이터 없음")

    # ── 전략 A ──────────────────────────────────────────────────────────
    dd_th = args.dd_th / 100
    dd_ok = dd >= dd_th
    rsi_ok = last["RSI"] <= args.rsi_th
    vix_ok = last["VIX"] >= args.vix_th
    trigger_price_a = peak * (1 - dd_th)

    print("\n" + "-" * 60)
    print(f"전략A (신호 진입: 하락 {args.dd_th:.0f}%+ / RSI {args.rsi_th:.0f} 이하 / VIX {args.vix_th:.0f} 이상)")
    print("-" * 60)
    print(f"  하락률 조건 : {'충족' if dd_ok else '미충족'}  (하락 {args.dd_th:.0f}% 되는 가격 = ${trigger_price_a:.2f})")
    print(f"  RSI 조건    : {'충족' if rsi_ok else '미충족'}  (현재 {last['RSI']:.1f})")
    print(f"  VIX 조건    : {'충족' if vix_ok else '미충족'}  (현재 {last['VIX']:.1f})")
    if dd_ok and rsi_ok and vix_ok:
        print(f"  => 지금 3개 조건이 모두 충족된 상태입니다. (오늘 종가 ${last['QQQ']:.2f} 기준)")
    else:
        missing = []
        if not dd_ok:
            missing.append(f"QQQ가 ${trigger_price_a:.2f} 이하로 더 하락")
        if not rsi_ok:
            missing.append(f"RSI가 {args.rsi_th:.0f} 이하로 하락")
        if not vix_ok:
            missing.append(f"VIX가 {args.vix_th:.0f} 이상으로 상승")
        print(f"  => 아직 조건 미충족. 필요: {' / '.join(missing)} (3개 동시 충족 필요)")

    # ── 전략 B ──────────────────────────────────────────────────────────
    rebound_pct = args.rebound_pct / 100
    pullback_pct = args.pullback_pct / 100
    breakout_buffer = args.breakout_buffer / 100
    phase, episode_low, swing_high, last_ath_date = replay_breakout_phase(
        df["QQQ"], dd_th, rebound_pct, pullback_pct)

    print("\n" + "-" * 60)
    print("전략B (매물대 돌파)")
    print("-" * 60)
    print(f"  기준 전고점(직전 사상최고가) : {last_ath_date.date().isoformat()}")
    print(f"  현재 국면(phase)            : {phase}")
    if phase == "IDLE":
        print(f"  => 아직 {args.dd_th:.0f}% 이상 하락이 나오지 않아 하락 국면이 시작되지 않았습니다.")
        print(f"     (QQQ가 ${peak*(1-dd_th):.2f} 이하로 떨어지면 하락 국면 시작)")
    elif phase == "DECLINE":
        print(f"  현재까지 저점                : ${episode_low:.2f}")
        need = episode_low * (1 + rebound_pct)
        print(f"  => 아직 저점 대비 반등이 부족합니다. QQQ가 ${need:.2f} 이상 반등하면 매물대 추적 시작.")
    elif phase == "REBOUND":
        print(f"  현재 추적 중인 반등고점(매물대 후보) : ${swing_high:.2f}")
        need = swing_high * (1 - pullback_pct)
        print(f"  => 아직 매물대가 확정되지 않았습니다. QQQ가 ${need:.2f} 이하로 눌림목이 나오면 매물대 확정.")
    elif phase == "CONFIRMED":
        entry_price = swing_high * (1 + breakout_buffer)
        print(f"  확정된 매물대(반등고점)        : ${swing_high:.2f}")
        print(f"  => QQQ가 ${entry_price:.2f} 이상으로 종가 마감하면 진입 신호! (매물대 대비 +{args.breakout_buffer:.2f}%)")
        if last["QQQ"] >= entry_price:
            print(f"  => 오늘 종가(${last['QQQ']:.2f})가 이미 돌파가를 넘었습니다. 진입 조건 충족!")

    print("\n" + "=" * 60)
    print("※ 위 내용은 신규 진입(매수) 조건만 확인합니다. 이미 TQQQ/BULZ를 보유 중이라면")
    print("  매도 조건(RSI 60 도달 후 주봉 2주 연속 하락)은 local_compare_backtest.py 의")
    print("  거래 내역으로 확인하세요. 투자 조언이 아니며 과거 로직을 기계적으로 계산한")
    print("  결과일 뿐입니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
