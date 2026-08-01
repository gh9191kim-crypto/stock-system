"""
전략C(기본 레버리지 슬리브 + 나머지 래칫) 그리드 서치.

"조정 없는 상승장에서도 레버리지를 놓치지 않으면서, 최대낙폭(MDD)은 -50%를
넘기지 않는 조합"을 찾기 위해 여러 파라미터 조합을 한 번에 백테스트하고,
MDD -50% 이내인 조합 중 수익률이 가장 높은 순으로 정렬해서 보여준다.

사용법:
    pip install yfinance pandas numpy
    python3 grid_search_hybrid.py
    python3 grid_search_hybrid.py --mdd-limit 50 --seed-krw 600000000
    python3 grid_search_hybrid.py --start 2019-01-01 --end 2026-07-31
"""
import argparse
import itertools
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
import compare as engine  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="전략C(기본 레버리지+래칫) 그리드 서치")
    p.add_argument("--seed-krw", type=float, default=600_000_000)
    p.add_argument("--start", type=str, default="2019-01-01")
    p.add_argument("--end", type=str, default="2026-07-31")
    p.add_argument("--fee-pct", type=float, default=0.1)
    p.add_argument("--no-fx", action="store_true")
    p.add_argument("--mdd-limit", type=float, default=50.0,
                    help="이 값(%%)보다 낙폭이 깊은 조합은 제외 (기본 50)")
    p.add_argument("--top", type=int, default=15, help="상위 몇 개를 출력할지")
    p.add_argument("--out", type=str, default="grid_search_result.csv")

    # 그리드 범위 (기본값도 조정 가능하게 열어둠)
    p.add_argument("--base-grid", type=str, default="0,10,20,30,40,50",
                    help="기본 레버리지 비중(%%) 후보, 쉼표구분")
    p.add_argument("--th4-grid", type=str, default="25,30,35",
                    help="4차(전량 TQQQ) 전환 하락률(%%) 후보")
    p.add_argument("--w1-grid", type=str, default="20,30,40", help="1차 비중 QLD(%%) 후보")
    p.add_argument("--w2-grid", type=str, default="20,30,40", help="2차 비중 TQQQ(%%) 후보")
    return p.parse_args()


def fnum(s):
    return [float(x) for x in s.split(",")]


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
    apply_fx = not args.no_fx
    fee_rate = args.fee_pct / 100
    mdd_limit = -abs(args.mdd_limit) / 100  # 예: -0.50

    base_grid = fnum(args.base_grid)
    th4_grid = fnum(args.th4_grid)
    w1_grid = fnum(args.w1_grid)
    w2_grid = fnum(args.w2_grid)

    print(f"시세 데이터 조회 중... ({args.start} ~ {args.end})")
    end_exclusive = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = engine.fetch_prices(args.start, end_exclusive)
    except Exception as e:
        print(f"\n[오류] 시세 데이터를 불러오지 못했습니다: {e}")
        sys.exit(1)
    if df.empty or len(df) < 30:
        print("\n[오류] 해당 기간의 시세 데이터가 부족합니다.")
        sys.exit(1)
    print(f"데이터 {len(df)}거래일 확보.\n")

    combos = list(itertools.product(base_grid, w1_grid, w2_grid, th4_grid))
    print(f"총 {len(combos)}개 조합 백테스트 실행 중...")

    results = []
    for base, w1, w2, th4 in combos:
        if w1 + w2 >= 100:
            continue  # 3차 단계 QQQ 비중이 음수가 되는 불가능한 조합 제외
        th = {"th1": 0.10, "th2": 0.15, "th3": 0.20, "th4": th4 / 100}
        w = {"w1": w1 / 100, "w2": w2 / 100}
        r, events, _ = engine.run_hybrid_ratchet_strategy(
            df, args.seed_krw, base / 100, th, w, apply_fx, fee_rate)
        dates = [rec["date"] for rec in r]
        values = [rec["value_krw"] for rec in r]
        m = engine.calc_metrics(dates, values)
        results.append({
            "base_pct": base, "w1": w1, "w2": w2, "th4": th4,
            "final_krw": values[-1], "total_return": m["total_return"],
            "cagr": m["cagr"], "mdd": m["mdd"], "sharpe": m["sharpe"],
            "trades": len(events),
        })

    all_df = pd.DataFrame(results)
    all_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"전체 {len(all_df)}개 조합 결과를 {args.out} 로 저장했습니다.\n")

    passed = all_df[all_df["mdd"] >= mdd_limit].copy()
    passed = passed.sort_values("total_return", ascending=False)

    print("=" * 90)
    print(f"MDD -{args.mdd_limit:.0f}% 이내 조합: {len(passed)} / {len(all_df)}개 통과")
    print("=" * 90)
    if passed.empty:
        print("조건을 만족하는 조합이 없습니다. --mdd-limit 값을 높여보세요.")
    else:
        show = passed.head(args.top).copy()
        show["final_krw"] = show["final_krw"].map(lambda v: f"{v:,.0f}원")
        show["total_return"] = show["total_return"].map(lambda v: f"{v*100:.1f}%")
        show["cagr"] = show["cagr"].map(lambda v: "-" if v is None else f"{v*100:.1f}%")
        show["mdd"] = show["mdd"].map(lambda v: f"{v*100:.1f}%")
        show["sharpe"] = show["sharpe"].map(lambda v: "-" if v is None else f"{v:.2f}")
        print(show.rename(columns={
            "base_pct": "기본QLD%", "w1": "1차비중%", "w2": "2차비중%", "th4": "4차임계%",
            "final_krw": "최종금액", "total_return": "총수익률", "cagr": "CAGR",
            "mdd": "MDD", "sharpe": "샤프", "trades": "전환횟수",
        }).to_string(index=False))

        best = passed.iloc[0]
        print("\n" + "-" * 90)
        print("최고 수익 조합 (MDD 제한 내):")
        print(f"  기본 QLD 비중 {best['base_pct']:.0f}% / 1차전환 {best['w1']:.0f}% / "
              f"2차전환 {best['w2']:.0f}% / 4차임계 -{best['th4']:.0f}%")
        print(f"  최종금액 {best['final_krw']:,.0f}원 / 총수익률 {best['total_return']*100:.1f}% / "
              f"CAGR {best['cagr']*100:.1f}% / MDD {best['mdd']*100:.1f}% / 전환 {int(best['trades'])}회")

    # ── 참고 벤치마크 ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("참고 벤치마크")
    print("=" * 90)
    bench_qqq = engine.run_benchmark(df, args.seed_krw, apply_fx)
    bench_qld = engine.run_buyhold(df, args.seed_krw, "QLD", apply_fx)
    bench_tqqq = engine.run_buyhold(df, args.seed_krw, "TQQQ", apply_fx)
    dates = [x["date"] for x in bench_qqq]
    for label, series in [("QQQ 100% 보유", bench_qqq), ("QLD 100% 매수후보유", bench_qld),
                           ("TQQQ 100% 매수후보유", bench_tqqq)]:
        vals = [x["value_krw"] for x in series]
        m = engine.calc_metrics(dates, vals)
        print(f"  {label:<20s}: 최종 {vals[-1]:,.0f}원 / 수익률 {m['total_return']*100:.1f}% / "
              f"CAGR {m['cagr']*100:.1f}% / MDD {m['mdd']*100:.1f}%")

    th_default = {"th1": 0.10, "th2": 0.15, "th3": 0.20, "th4": 0.30}
    w_default = {"w1": 0.30, "w2": 0.30}
    r0, e0, _ = engine.run_ratchet_strategy(df, args.seed_krw, th_default, w_default, apply_fx, fee_rate)
    vals0 = [x["value_krw"] for x in r0]
    m0 = engine.calc_metrics(dates, vals0)
    print(f"  {'기존 래칫(base 0%, 기본설정)':<20s}: 최종 {vals0[-1]:,.0f}원 / 수익률 {m0['total_return']*100:.1f}% / "
          f"CAGR {m0['cagr']*100:.1f}% / MDD {m0['mdd']*100:.1f}% / 전환 {len(e0)}회")

    # ── 위기 국면별 손실률 비교 (최고 조합 vs 기존 래칫 vs 벤치마크) ──────
    if not passed.empty:
        best_th = {"th1": 0.10, "th2": 0.15, "th3": 0.20, "th4": best["th4"] / 100}
        best_w = {"w1": best["w1"] / 100, "w2": best["w2"] / 100}
        r_best, e_best, _ = engine.run_hybrid_ratchet_strategy(
            df, args.seed_krw, best["base_pct"] / 100, best_th, best_w, apply_fx, fee_rate)
        vals_best = [x["value_krw"] for x in r_best]

        series_for_events = [
            ("QQQ 100% 보유", [x["value_krw"] for x in bench_qqq], []),
            ("QLD 100% 매수후보유", [x["value_krw"] for x in bench_qld], []),
            ("TQQQ 100% 매수후보유", [x["value_krw"] for x in bench_tqqq], []),
            ("기존 래칫(base 0%)", vals0, e0),
            (f"최고조합(QLD{best['base_pct']:.0f}%+래칫)", vals_best, e_best),
        ]

        print("\n" + "=" * 90)
        print("위기 국면별 손실률/수익률 비교 (구간 날짜는 일반 지식 기반 근사치)")
        print("=" * 90)
        for win in engine.EVENT_WINDOWS:
            print(f"\n[{win['name']}] {win['start']} ~ {win['end']}")
            for label, vals, evs in series_for_events:
                start_v = find_value_near(dates, vals, win["start"], "first")
                end_v = find_value_near(dates, vals, win["end"], "last")
                if start_v is None or end_v is None:
                    print(f"  - {label}: 데이터 범위 밖")
                    continue
                ret = (end_v / start_v - 1) * 100
                entered = any(win["start"] <= ev["date"] <= win["end"] for ev in evs)
                mark = " [전환 발생]" if entered else ""
                print(f"  - {label}: {ret:+.1f}%{mark}")


if __name__ == "__main__":
    main()
