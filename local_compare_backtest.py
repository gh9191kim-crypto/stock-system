"""
로컬에서 바로 실행하는 전략 비교 백테스트 (Vercel 배포 없이 결과 확인용)

api/compare.py 와 완전히 동일한 로직을 그대로 불러와 사용하므로, 여기서 나온
숫자는 나중에 웹(compare.html)에서 같은 파라미터로 돌렸을 때와 동일합니다.

사용법:
    pip install yfinance pandas numpy
    python3 local_compare_backtest.py
    python3 local_compare_backtest.py --seed-krw 600000000 --start 2019-01-01 --end 2026-07-31
    python3 local_compare_backtest.py --rsi-exit-th 60 --down-weeks 2

결과는 콘솔에 표로 출력되고, 일별 시계열은 backtest_result.csv 로 저장됩니다.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import compare as engine  # noqa: E402

import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="나스닥 하락 대응 전략 비교 백테스트 (로컬 실행)")
    p.add_argument("--seed-krw", type=float, default=600_000_000)
    p.add_argument("--start", type=str, default="2019-01-01")
    p.add_argument("--end", type=str, default="2026-07-31")
    p.add_argument("--fee-pct", type=float, default=0.1, help="전환 시 수수료/슬리피지 (%%)")
    p.add_argument("--no-fx", action="store_true", help="환율 변동을 반영하지 않고 시작일 환율로 고정")

    p.add_argument("--dd-th", type=float, default=10.0, help="전략A 진입 하락률 기준(%%)")
    p.add_argument("--rsi-th", type=float, default=30.0, help="전략A 진입 RSI 이하 기준")
    p.add_argument("--vix-th", type=float, default=20.0, help="전략A 진입 VIX 이상 기준")

    p.add_argument("--rebound-pct", type=float, default=5.0, help="전략B 반등 확인(저점대비 %%)")
    p.add_argument("--pullback-pct", type=float, default=3.0, help="전략B 매물대 확정 눌림목(%%)")
    p.add_argument("--breakout-buffer", type=float, default=0.5, help="전략B 돌파 확인 버퍼(%%)")

    p.add_argument("--rsi-exit-th", type=float, default=60.0, help="매도 대기 해제 RSI 기준 (A/B 공통)")
    p.add_argument("--down-weeks", type=int, default=2, help="주봉 연속 하락 매도 기준(주)")

    p.add_argument("--th1", type=float, default=10.0, help="이전 전략(래칫) 1차 QLD 전환 하락률(%%)")
    p.add_argument("--th2", type=float, default=15.0, help="이전 전략 2차 TQQQ 전환 하락률(%%)")
    p.add_argument("--th3", type=float, default=20.0, help="이전 전략 3차 잔여 TQQQ 전환 하락률(%%)")
    p.add_argument("--th4", type=float, default=30.0, help="이전 전략 4차 전량 TQQQ 전환 하락률(%%)")
    p.add_argument("--w1", type=float, default=30.0, help="이전 전략 1차 비중 QLD(%%)")
    p.add_argument("--w2", type=float, default=30.0, help="이전 전략 2차 비중 TQQQ(%%)")

    p.add_argument("--out", type=str, default="backtest_result.csv", help="일별 결과 CSV 저장 경로")
    return p.parse_args()


def fmt_krw(v):
    return f"{v:,.0f}원"


def fmt_pct(v):
    return "-" if v is None else f"{v*100:.1f}%"


SERIES_META = [
    ("benchmark_qqq", "QQQ 100% 보유"),
    ("ratchet", "이전 전략 (단계적 전환)"),
    ("signal_tqqq", "전략A 신호진입 - TQQQ"),
    ("signal_bulz", "전략A 신호진입 - BULZ"),
    ("breakout_tqqq", "전략B 매물대돌파 - TQQQ"),
    ("breakout_bulz", "전략B 매물대돌파 - BULZ"),
]


def find_value_near(dates, values, target, mode):
    if mode == "first":
        for d, v in zip(dates, values):
            if d >= target:
                return v
    else:
        for d, v in zip(reversed(dates), reversed(values)):
            if d <= target:
                return v
    return None


def main():
    args = parse_args()

    print(f"시세 데이터 조회 중... ({args.start} ~ {args.end})")
    end_exclusive = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = engine.fetch_prices(args.start, end_exclusive)
    except Exception as e:
        print(f"\n[오류] 시세 데이터를 불러오지 못했습니다: {e}")
        print("인터넷 연결 및 yfinance 설치 상태를 확인해주세요 (pip install -U yfinance).")
        sys.exit(1)

    if df.empty or len(df) < 30:
        print("\n[오류] 해당 기간의 시세 데이터가 부족합니다. 기간을 조정해주세요.")
        sys.exit(1)

    print(f"데이터 {len(df)}거래일 확보. 백테스트 실행 중...\n")

    apply_fx = not args.no_fx
    fee_rate = args.fee_pct / 100
    dd_th = args.dd_th / 100
    rebound_pct = args.rebound_pct / 100
    pullback_pct = args.pullback_pct / 100
    breakout_buffer = args.breakout_buffer / 100
    th = {"th1": args.th1 / 100, "th2": args.th2 / 100, "th3": args.th3 / 100, "th4": args.th4 / 100}
    w = {"w1": args.w1 / 100, "w2": args.w2 / 100}

    exit_days = engine.weekly_down_streak_exit_days(df["QQQ"], args.down_weeks)

    series, events = {}, {}
    r, e, _ = engine.run_ratchet_strategy(df, args.seed_krw, th, w, apply_fx, fee_rate)
    series["ratchet"], events["ratchet"] = r, e

    series["benchmark_qqq"] = engine.run_benchmark(df, args.seed_krw, apply_fx)
    events["benchmark_qqq"] = []

    r, e, _ = engine.run_signal_strategy(df, args.seed_krw, "TQQQ", dd_th, args.rsi_th, args.vix_th,
                                          exit_days, args.rsi_exit_th, apply_fx, fee_rate)
    series["signal_tqqq"], events["signal_tqqq"] = r, e

    r, e, _ = engine.run_signal_strategy(df, args.seed_krw, "BULZ", dd_th, args.rsi_th, args.vix_th,
                                          exit_days, args.rsi_exit_th, apply_fx, fee_rate)
    series["signal_bulz"], events["signal_bulz"] = r, e

    r, e, _ = engine.run_breakout_strategy(df, args.seed_krw, "TQQQ", dd_th, rebound_pct, pullback_pct,
                                            breakout_buffer, exit_days, args.rsi_exit_th, apply_fx, fee_rate)
    series["breakout_tqqq"], events["breakout_tqqq"] = r, e

    r, e, _ = engine.run_breakout_strategy(df, args.seed_krw, "BULZ", dd_th, rebound_pct, pullback_pct,
                                            breakout_buffer, exit_days, args.rsi_exit_th, apply_fx, fee_rate)
    series["breakout_bulz"], events["breakout_bulz"] = r, e

    dates = [rec["date"] for rec in series["benchmark_qqq"]]
    value_series = {k: [rec["value_krw"] for rec in v] for k, v in series.items()}
    metrics = {k: engine.calc_metrics(dates, v) for k, v in value_series.items()}

    bulz_valid = df["BULZ"].dropna()
    bulz_inception = bulz_valid.index[0].date().isoformat() if len(bulz_valid) else None

    # ── 요약 표 ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("전략별 최종 성과 비교")
    print("=" * 78)
    rows = []
    for key, label in SERIES_META:
        m = metrics[key]
        final_v = value_series[key][-1]
        rows.append({
            "전략": label,
            "최종평가금액": fmt_krw(final_v),
            "총수익률": fmt_pct(m["total_return"]),
            "CAGR": fmt_pct(m["cagr"]),
            "MDD": fmt_pct(m["mdd"]),
            "샤프": "-" if m["sharpe"] is None else f"{m['sharpe']:.2f}",
            "매매횟수": len(events[key]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 위기 국면별 비교 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("위기 국면별 비교 (구간 날짜는 일반 지식 기반 근사치)")
    print("=" * 78)
    for win in engine.EVENT_WINDOWS:
        print(f"\n[{win['name']}] {win['start']} ~ {win['end']}")
        for key, label in SERIES_META:
            start_v = find_value_near(dates, value_series[key], win["start"], "first")
            end_v = find_value_near(dates, value_series[key], win["end"], "last")
            if start_v is None or end_v is None:
                print(f"  - {label}: 데이터 범위 밖")
                continue
            ret = (end_v / start_v - 1) * 100
            entered = any(ev["type"] == "BUY" and win["start"] <= ev["date"] <= win["end"]
                          for ev in events[key])
            mark = " [진입]" if entered else ""
            print(f"  - {label}: {ret:+.1f}%{mark}")

    # ── 거래 내역 (요약) ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("전략별 거래 내역")
    print("=" * 78)
    for key, label in SERIES_META:
        evs = events[key]
        if not evs:
            continue
        print(f"\n[{label}] 총 {len(evs)}건")
        for ev in evs:
            print(f"  {ev['date']}  {ev['type']:5s}  {ev['reason']:<45s}  ${ev['value_usd']:,.0f}")

    if bulz_inception:
        print(f"\n※ BULZ 데이터는 {bulz_inception} 부터 존재합니다. 그 이전 신호는 QQQ 보유로 대체됩니다.")

    # ── CSV 저장 ────────────────────────────────────────────────────────
    out_df = pd.DataFrame({"date": dates, **{k: value_series[k] for k, _ in SERIES_META}})
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n일별 결과를 {args.out} 로 저장했습니다.")


if __name__ == "__main__":
    main()
