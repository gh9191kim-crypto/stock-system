"""
Vercel Python Serverless Function — 전략 비교 백테스트
GET /api/compare?seed_krw=...&start=2019-01-01&end=2026-07-31&...

기존 "단계적 하락 대응 전환"(래칫) 전략과, 신호 기반 두 가지 진입 전략을
같은 기간·같은 시드로 동시에 계산해 비교한다.

전략 A (신호 기반): QQQ 고점대비 하락률 >= dd_th AND RSI(14) <= rsi_th
                    AND VIX >= vix_th  ->  TQQQ 또는 BULZ 전액 진입
전략 B (매물대 돌파): 고점대비 하락률 >= dd_th 로 하락 국면 시작
                     -> 저점 대비 rebound_pct 이상 반등하면 반등고점(매물대) 추적 시작
                     -> 반등고점 대비 pullback_pct 이상 눌림목이 나오면 매물대 '확정'
                     -> 확정된 매물대를 breakout_buffer 만큼 상향 돌파하면 TQQQ/BULZ 진입
                     (돌파 전 저점을 재이탈하면 하락 국면으로 리셋)
매도 규칙 (A/B 공통, 2단계): 매수 직후에는 RSI(14)가 rsi_exit_th(기본 60) 이상으로
                    올라가기 전까지는 매도 신호를 무시하고 보유를 유지한다. RSI가
                    rsi_exit_th 이상에 도달한 뒤부터는 주봉 종가가 down_weeks
                    (기본 2주) 연속 하락 확정 시 QQQ로 전량 복귀한다.
두 전략 모두 TQQQ / BULZ 버전을 각각 별도로 계산해 총 4개 시리즈를 만들고,
기존 래칫 전략(QQQ->QLD->TQQQ 단계적 비가역 전환) 및 QQQ 100% 단순보유와 함께 반환한다.
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["QQQ", "QLD", "TQQQ", "BULZ", "^VIX", "KRW=X"]

# 사용자가 언급한 위기 국면의 대략적인 날짜 구간 (일반적으로 알려진 사실 기준 근사치 —
# 이 세션에서 실제 시세 데이터로 검증하지 못했으므로 참고용이며, 실제 수치는 아래
# 계산된 시계열에서 해당 구간을 그대로 슬라이스해 산출한다)
EVENT_WINDOWS = [
    {"name": "COVID-19 팬데믹 폭락", "start": "2020-02-19", "end": "2020-08-01"},
    {"name": "2022 금리인상·인플레이션 약세장", "start": "2022-01-03", "end": "2022-12-30"},
    {"name": "러시아-우크라이나 전쟁 발발", "start": "2022-02-24", "end": "2022-04-30"},
    {"name": "2025 관세(Liberation Day) 충격", "start": "2025-04-02", "end": "2025-05-15"},
    {"name": "이란-이스라엘-미국 분쟁 (2025.6)", "start": "2025-06-13", "end": "2025-07-15"},
]


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw[TICKERS].rename(columns={"^VIX": "VIX"})
    core_cols = ["QQQ", "QLD", "TQQQ", "VIX", "KRW=X"]
    core = raw[core_cols].ffill().dropna(how="any")
    bulz = raw["BULZ"].ffill()
    df = core.join(bulz)
    df["RSI"] = compute_rsi(df["QQQ"], 14)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def weekly_down_streak_exit_days(qqq_close: pd.Series, weeks_needed: int = 2) -> set:
    weekly = qqq_close.resample("W-FRI").last().dropna()
    down = weekly.diff() < 0
    streak = down.groupby((~down).cumsum()).cumcount() + 1
    streak = streak.where(down, 0)
    exit_weeks = weekly.index[streak >= weeks_needed]
    idx = qqq_close.index
    exit_days = set()
    for wf in exit_weeks:
        pos = idx.searchsorted(wf, side="right") - 1
        if pos >= 0:
            exit_days.add(idx[pos])
    return exit_days


def fx_series_for(df: pd.DataFrame, apply_fx: bool) -> pd.Series:
    fx0 = df["KRW=X"].iloc[0]
    return df["KRW=X"] if apply_fx else pd.Series(fx0, index=df.index)


# ── 전략 A: RSI/VIX/하락률 신호 진입 ──────────────────────────────────────
def run_signal_strategy(df, capital_krw, leverage_col, dd_th, rsi_th, vix_th,
                         exit_days, rsi_exit_arm_th, apply_fx, fee_rate):
    fx_series = fx_series_for(df, apply_fx)
    fx0 = fx_series.iloc[0]
    capital_usd = capital_krw / fx0
    shares = {"QQQ": capital_usd / df["QQQ"].iloc[0], "LEV": 0.0}
    peak = df["QQQ"].iloc[0]
    in_position = False
    armed = False  # 매수 후 RSI가 rsi_exit_arm_th 이상 도달했는지 (도달 전에는 매도 신호 무시)
    records, events = [], []
    total_fee = 0.0

    for dt, row in df.iterrows():
        qqq = row["QQQ"]
        peak = max(peak, qqq)
        dd = (peak - qqq) / peak
        lev_px = row[leverage_col]
        rsi = row["RSI"]

        if in_position:
            if not armed and pd.notna(rsi) and rsi >= rsi_exit_arm_th:
                armed = True
            if armed and dt in exit_days:
                value = shares["QQQ"] * qqq + shares["LEV"] * lev_px
                fee = value * fee_rate
                value -= fee
                total_fee += fee
                shares = {"QQQ": value / qqq, "LEV": 0.0}
                events.append({"date": dt.date().isoformat(), "type": "SELL",
                                "reason": f"RSI {rsi_exit_arm_th:.0f} 이상 도달 후 주봉 2주 연속 하락",
                                "value_usd": value})
                in_position = False
                armed = False
        elif dd >= dd_th and pd.notna(rsi) and rsi <= rsi_th \
                and row["VIX"] >= vix_th and pd.notna(lev_px):
            value = shares["QQQ"] * qqq
            fee = value * fee_rate
            value -= fee
            total_fee += fee
            shares = {"QQQ": 0.0, "LEV": value / lev_px}
            events.append({"date": dt.date().isoformat(), "type": "BUY",
                            "reason": f"QQQ -{dd*100:.1f}%, RSI {rsi:.1f}, VIX {row['VIX']:.1f}",
                            "value_usd": value})
            in_position = True
            armed = False

        value_usd = shares["QQQ"] * qqq + (shares["LEV"] * lev_px if pd.notna(lev_px) else 0.0)
        fx = fx_series.loc[dt]
        records.append({"date": dt.date().isoformat(), "value_krw": value_usd * fx})

    return records, events, total_fee


# ── 전략 B: 하락 저점 후 반등고점(매물대) 돌파 ────────────────────────────
def run_breakout_strategy(df, capital_krw, leverage_col, dd_th, rebound_pct, pullback_pct,
                           breakout_buffer, exit_days, rsi_exit_arm_th, apply_fx, fee_rate):
    fx_series = fx_series_for(df, apply_fx)
    fx0 = fx_series.iloc[0]
    capital_usd = capital_krw / fx0
    shares = {"QQQ": capital_usd / df["QQQ"].iloc[0], "LEV": 0.0}
    peak = df["QQQ"].iloc[0]
    in_position = False
    armed = False  # 매수 후 RSI가 rsi_exit_arm_th 이상 도달했는지 (도달 전에는 매도 신호 무시)
    phase = "IDLE"
    episode_low = None
    swing_high = None
    records, events = [], []
    total_fee = 0.0

    for dt, row in df.iterrows():
        qqq = row["QQQ"]
        peak = max(peak, qqq)
        dd = (peak - qqq) / peak
        lev_px = row[leverage_col]
        rsi = row["RSI"]

        if in_position:
            if not armed and pd.notna(rsi) and rsi >= rsi_exit_arm_th:
                armed = True
            if armed and dt in exit_days:
                value = shares["QQQ"] * qqq + shares["LEV"] * lev_px
                fee = value * fee_rate
                value -= fee
                total_fee += fee
                shares = {"QQQ": value / qqq, "LEV": 0.0}
                events.append({"date": dt.date().isoformat(), "type": "SELL",
                                "reason": f"RSI {rsi_exit_arm_th:.0f} 이상 도달 후 주봉 2주 연속 하락",
                                "value_usd": value})
                in_position = False
                armed = False
                phase, episode_low, swing_high = "IDLE", None, None
        else:
            if phase == "IDLE":
                if dd >= dd_th:
                    phase, episode_low = "DECLINE", qqq
            elif phase == "DECLINE":
                episode_low = min(episode_low, qqq)
                if qqq >= episode_low * (1 + rebound_pct):
                    phase, swing_high = "REBOUND", qqq
            elif phase == "REBOUND":
                if qqq < episode_low:
                    episode_low, phase, swing_high = qqq, "DECLINE", None
                else:
                    swing_high = max(swing_high, qqq)
                    if qqq <= swing_high * (1 - pullback_pct):
                        phase = "CONFIRMED"
            elif phase == "CONFIRMED":
                if qqq < episode_low:
                    episode_low, phase, swing_high = qqq, "DECLINE", None
                elif qqq >= swing_high * (1 + breakout_buffer) and pd.notna(lev_px):
                    value = shares["QQQ"] * qqq
                    fee = value * fee_rate
                    value -= fee
                    total_fee += fee
                    shares = {"QQQ": 0.0, "LEV": value / lev_px}
                    events.append({"date": dt.date().isoformat(), "type": "BUY",
                                    "reason": f"매물대(${swing_high:.2f}) 상향돌파(${qqq:.2f})",
                                    "value_usd": value})
                    in_position = True
                    armed = False
                    phase, episode_low, swing_high = "IDLE", None, None

        value_usd = shares["QQQ"] * qqq + (shares["LEV"] * lev_px if pd.notna(lev_px) else 0.0)
        fx = fx_series.loc[dt]
        records.append({"date": dt.date().isoformat(), "value_krw": value_usd * fx})

    return records, events, total_fee


# ── 기존 래칫 전략 (QQQ -> QLD -> TQQQ 단계적 비가역 전환) ─────────────────
def ratchet_target_weights(w1, w2):
    return [
        {"QQQ": 1.0, "QLD": 0.0, "TQQQ": 0.0},
        {"QQQ": 1 - w1, "QLD": w1, "TQQQ": 0.0},
        {"QQQ": 1 - w1 - w2, "QLD": w1, "TQQQ": w2},
        {"QQQ": 0.0, "QLD": w1, "TQQQ": 1 - w1},
        {"QQQ": 0.0, "QLD": 0.0, "TQQQ": 1.0},
    ]


def run_ratchet_strategy(df, capital_krw, th, w, apply_fx, fee_rate):
    assets = ["QQQ", "QLD", "TQQQ"]
    fx_series = fx_series_for(df, apply_fx)
    fx0 = fx_series.iloc[0]
    capital_usd = capital_krw / fx0
    shares = {"QQQ": capital_usd / df["QQQ"].iloc[0], "QLD": 0.0, "TQQQ": 0.0}
    target_states = ratchet_target_weights(w["w1"], w["w2"])
    peak = df["QQQ"].iloc[0]
    state = 0
    records, events = [], []
    total_fee = 0.0

    for dt, row in df.iterrows():
        peak = max(peak, row["QQQ"])
        dd = (peak - row["QQQ"]) / peak
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
            target_w = target_states[new_state]
            value = sum(shares[a] * row[a] for a in assets)
            turnover = sum(abs(value * target_w[a] - shares[a] * row[a]) for a in assets) / 2
            fee = turnover * fee_rate
            value -= fee
            total_fee += fee
            shares = {a: (value * target_w[a]) / row[a] if target_w[a] > 0 else 0.0 for a in assets}
            events.append({"date": dt.date().isoformat(), "type": "REBALANCE",
                            "reason": f"QQQ -{dd*100:.1f}% (단계 {new_state})", "value_usd": value})
            state = new_state
        value_usd = sum(shares[a] * row[a] for a in assets)
        fx = fx_series.loc[dt]
        records.append({"date": dt.date().isoformat(), "value_krw": value_usd * fx})

    return records, events, total_fee


def run_buyhold(df, capital_krw, ticker, apply_fx):
    """ticker를 시작일에 전액 매수해 그대로 보유 (하락 대응 없이 단순 매수후보유).
    QQQ/QLD/TQQQ 처럼 조회 구간 전체에 데이터가 있는 티커에만 사용한다 (BULZ처럼
    중간에 상장한 티커를 넣으면 시작일 데이터가 없어 날짜 배열 길이가 달라진다)."""
    fx_series = fx_series_for(df, apply_fx)
    fx0 = fx_series.iloc[0]
    capital_usd = capital_krw / fx0
    shares = capital_usd / df[ticker].iloc[0]
    records = []
    for dt, row in df.iterrows():
        value_usd = shares * row[ticker]
        records.append({"date": dt.date().isoformat(), "value_krw": value_usd * fx_series.loc[dt]})
    return records


def run_benchmark(df, capital_krw, apply_fx):
    return run_buyhold(df, capital_krw, "QQQ", apply_fx)


def calc_metrics(dates, values):
    s = pd.Series(values, index=pd.to_datetime(dates))
    ret = s.pct_change().dropna()
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    total_return = s.iloc[-1] / s.iloc[0] - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else None
    running_max = s.cummax()
    mdd = (s / running_max - 1).min()
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol else None
    return {
        "total_return": float(total_return),
        "cagr": float(cagr) if cagr is not None else None,
        "mdd": float(mdd),
        "vol": float(vol),
        "sharpe": float(sharpe) if sharpe is not None else None,
    }


def sanitize(obj):
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


class handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict):
        body = json.dumps(sanitize(payload)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "s-maxage=1800, stale-while-revalidate=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)

            def qf(key, default):
                return float(q.get(key, [default])[0])

            seed_krw = qf("seed_krw", 600_000_000)
            start = q.get("start", ["2019-01-01"])[0]
            end = q.get("end", ["2026-07-31"])[0]
            apply_fx = q.get("fx", ["1"])[0] != "0"
            fee_rate = qf("fee", 0.1) / 100
            down_weeks = int(qf("down_weeks", 2))

            dd_th = qf("dd_th", 10) / 100
            rsi_th = qf("rsi_th", 30)
            vix_th = qf("vix_th", 20)
            rsi_exit_th = qf("rsi_exit_th", 60)

            rebound_pct = qf("rebound_pct", 5) / 100
            pullback_pct = qf("pullback_pct", 3) / 100
            breakout_buffer = qf("breakout_buffer", 0.5) / 100

            th = {"th1": qf("th1", 10) / 100, "th2": qf("th2", 15) / 100,
                  "th3": qf("th3", 20) / 100, "th4": qf("th4", 30) / 100}
            w = {"w1": qf("w1", 30) / 100, "w2": qf("w2", 30) / 100}

            end_exclusive = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = fetch_prices(start, end_exclusive)
            if df.empty or len(df) < 30:
                self._send_json(400, {"error": "해당 기간의 시세 데이터가 부족합니다. 기간을 조정해주세요."})
                return

            exit_days = weekly_down_streak_exit_days(df["QQQ"], down_weeks)

            series, events, fees = {}, {}, {}

            r, e, f = run_ratchet_strategy(df, seed_krw, th, w, apply_fx, fee_rate)
            series["ratchet"], events["ratchet"], fees["ratchet"] = r, e, f

            bench = run_benchmark(df, seed_krw, apply_fx)
            series["benchmark_qqq"] = bench
            series["buyhold_tqqq"] = run_buyhold(df, seed_krw, "TQQQ", apply_fx)
            series["buyhold_qld"] = run_buyhold(df, seed_krw, "QLD", apply_fx)

            r, e, f = run_signal_strategy(df, seed_krw, "TQQQ", dd_th, rsi_th, vix_th, exit_days,
                                           rsi_exit_th, apply_fx, fee_rate)
            series["signal_tqqq"], events["signal_tqqq"], fees["signal_tqqq"] = r, e, f

            r, e, f = run_signal_strategy(df, seed_krw, "BULZ", dd_th, rsi_th, vix_th, exit_days,
                                           rsi_exit_th, apply_fx, fee_rate)
            series["signal_bulz"], events["signal_bulz"], fees["signal_bulz"] = r, e, f

            r, e, f = run_breakout_strategy(df, seed_krw, "TQQQ", dd_th, rebound_pct, pullback_pct,
                                             breakout_buffer, exit_days, rsi_exit_th, apply_fx, fee_rate)
            series["breakout_tqqq"], events["breakout_tqqq"], fees["breakout_tqqq"] = r, e, f

            r, e, f = run_breakout_strategy(df, seed_krw, "BULZ", dd_th, rebound_pct, pullback_pct,
                                             breakout_buffer, exit_days, rsi_exit_th, apply_fx, fee_rate)
            series["breakout_bulz"], events["breakout_bulz"], fees["breakout_bulz"] = r, e, f

            dates = [rec["date"] for rec in series["benchmark_qqq"]]
            value_series = {k: [rec["value_krw"] for rec in v] for k, v in series.items()}
            metrics = {k: calc_metrics(dates, v) for k, v in value_series.items()}

            bulz_inception = None
            bulz_valid = df["BULZ"].dropna()
            if len(bulz_valid):
                bulz_inception = bulz_valid.index[0].date().isoformat()

            qqq_running_max = df["QQQ"].cummax()
            qqq_dd = ((qqq_running_max - df["QQQ"]) / qqq_running_max).tolist()

            result = {
                "dates": dates,
                "qqq_dd": qqq_dd,
                "series": value_series,
                "events": events,
                "fees_usd": fees,
                "metrics": metrics,
                "event_windows": EVENT_WINDOWS,
                "bulz_inception": bulz_inception,
                "params": {
                    "dd_th": dd_th * 100, "rsi_th": rsi_th, "vix_th": vix_th, "down_weeks": down_weeks,
                    "rsi_exit_th": rsi_exit_th,
                    "rebound_pct": rebound_pct * 100, "pullback_pct": pullback_pct * 100,
                    "breakout_buffer": breakout_buffer * 100,
                    "ratchet": {"th1": th["th1"] * 100, "th2": th["th2"] * 100, "th3": th["th3"] * 100,
                                "th4": th["th4"] * 100, "w1": w["w1"] * 100, "w2": w["w2"] * 100},
                },
            }
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
