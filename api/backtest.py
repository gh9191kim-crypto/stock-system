"""
Vercel Python Serverless Function
GET /api/backtest?seed_krw=600000000&start=2019-01-01&end=2026-08-01
                  &th1=10&th2=15&th3=20&th4=30&w1=30&w2=30&fx=1&fee=0.1

나스닥(QQQ) 하락 대응 QLD/TQQQ 단계적 전환 백테스트를 실행하고 JSON으로 반환한다.
"""
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["QQQ", "QLD", "TQQQ", "KRW=X"]
ASSETS = ["QQQ", "QLD", "TQQQ"]


def fetch_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw[TICKERS].ffill().dropna(how="any")
    return raw


def target_weights(w1: float, w2: float):
    return [
        {"QQQ": 1.0, "QLD": 0.0, "TQQQ": 0.0},
        {"QQQ": 1 - w1, "QLD": w1, "TQQQ": 0.0},
        {"QQQ": 1 - w1 - w2, "QLD": w1, "TQQQ": w2},
        {"QQQ": 0.0, "QLD": w1, "TQQQ": 1 - w1},
        {"QQQ": 0.0, "QLD": 0.0, "TQQQ": 1.0},
    ]


def rebalance(shares: dict, row: pd.Series, target_w: dict, fee_rate: float):
    value = sum(shares[a] * row[a] for a in ASSETS)
    turnover = sum(abs(value * target_w[a] - shares[a] * row[a]) for a in ASSETS) / 2
    fee = turnover * fee_rate
    value_after_fee = value - fee
    new_shares = {a: (value_after_fee * target_w[a]) / row[a] if target_w[a] > 0 else 0.0 for a in ASSETS}
    return new_shares, fee


def run_backtest(prices: pd.DataFrame, capital_krw: float, th: dict, w: dict,
                  apply_fx: bool, fee_rate: float, benchmark_only: bool = False):
    df = prices.dropna(subset=ASSETS + ["KRW=X"]).copy()
    fx0 = df["KRW=X"].iloc[0]
    fx_series = df["KRW=X"] if apply_fx else pd.Series(fx0, index=df.index)

    capital_usd = capital_krw / fx0
    shares = {"QQQ": capital_usd / df["QQQ"].iloc[0], "QLD": 0.0, "TQQQ": 0.0}
    target_states = target_weights(w["w1"], w["w2"])

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
                shares, fee = rebalance(shares, row, target_states[new_state], fee_rate)
                total_fee_usd += fee
                v_after = sum(shares[a] * row[a] for a in ASSETS)
                events.append({
                    "date": dt.date().isoformat(), "drawdown": dd, "stage": new_state,
                    "w_QQQ": target_states[new_state]["QQQ"] * 100,
                    "w_QLD": target_states[new_state]["QLD"] * 100,
                    "w_TQQQ": target_states[new_state]["TQQQ"] * 100,
                    "value_usd": v_after, "fee_usd": fee,
                })
                state = new_state

        value_usd = sum(shares[a] * row[a] for a in ASSETS)
        fx = fx_series.loc[dt]
        records.append({
            "date": dt.date().isoformat(), "qqq_dd": dd, "value_krw": value_usd * fx,
            "w_QQQ": shares["QQQ"] * row["QQQ"] / value_usd if value_usd else 0.0,
            "w_QLD": shares["QLD"] * row["QLD"] / value_usd if value_usd else 0.0,
            "w_TQQQ": shares["TQQQ"] * row["TQQQ"] / value_usd if value_usd else 0.0,
        })

    return records, events, total_fee_usd


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
            end = q.get("end", [datetime.utcnow().strftime("%Y-%m-%d")])[0]
            th_pct = {"th1": qf("th1", 10), "th2": qf("th2", 15), "th3": qf("th3", 20), "th4": qf("th4", 30)}
            th = {k: v / 100 for k, v in th_pct.items()}
            w = {"w1": qf("w1", 30) / 100, "w2": qf("w2", 30) / 100}
            apply_fx = q.get("fx", ["1"])[0] != "0"
            fee_rate = qf("fee", 0.1) / 100

            end_exclusive = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            prices = fetch_prices(start, end_exclusive)
            if prices.empty or len(prices) < 2:
                self._send_json(400, {"error": "해당 기간의 시세 데이터가 부족합니다. 기간을 조정해주세요."})
                return

            records, events, total_fee = run_backtest(prices, seed_krw, th, w, apply_fx, fee_rate)
            bench_records, _, _ = run_backtest(prices, seed_krw, th, w, apply_fx, 0.0, benchmark_only=True)

            dates = [r["date"] for r in records]
            strat_krw = [r["value_krw"] for r in records]
            bench_krw = [b["value_krw"] for b in bench_records]

            result = {
                "dates": dates,
                "strategy": {
                    "value_krw": strat_krw,
                    "qqq_dd": [r["qqq_dd"] for r in records],
                    "w_QQQ": [r["w_QQQ"] for r in records],
                    "w_QLD": [r["w_QLD"] for r in records],
                    "w_TQQQ": [r["w_TQQQ"] for r in records],
                },
                "benchmark": {"value_krw": bench_krw},
                "events": events,
                "metrics": {
                    "strategy": calc_metrics(dates, strat_krw),
                    "benchmark": calc_metrics(dates, bench_krw),
                },
                "total_fee_usd": total_fee,
                "thresholds": th_pct,
            }
            self._send_json(200, result)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
