"""
필승매매 시스템 v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 백테스팅 엔진  (TQQQ 3중필터 / BULZ 전용모델)
② AI 자동매매 기록  (신호 발생 시 이유 설명)
③ 내 수동매매 입력 & 기록
④ AI vs 나 성과 비교
⑤ 매일 자동 업데이트 → 백테스트 재실행 → 모델 버전 관리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json, os, requests
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="필승매매 시스템", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#f5f7fa;}
.main .block-container{padding-top:1rem;}
div[data-testid="stSidebarContent"]{background:#fff;border-right:1px solid #e2e8f0;}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;
  padding:4px;border:1px solid #e2e8f0;gap:2px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:6px 14px;
  font-size:13px;color:#64748b;}
.stTabs [aria-selected="true"]{background:#1e40af!important;color:#fff!important;}

/* 카드 */
.kcard{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:14px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.kcard .lbl{font-size:10px;color:#94a3b8;text-transform:uppercase;
  letter-spacing:1.2px;margin-bottom:4px;}
.kcard .val{font-size:18px;font-weight:700;color:#1e293b;}
.kcard .sub{font-size:10px;color:#94a3b8;margin-top:2px;}

/* 신호 박스 */
.sig{border-radius:12px;padding:14px 18px;border-left:5px solid;margin-bottom:8px;}
.sig-buy {background:#f0fdf4;border-color:#16a34a;}
.sig-sell{background:#fff1f2;border-color:#dc2626;}
.sig-hold{background:#eff6ff;border-color:#2563eb;}

/* 색상 */
.up  {color:#15803d!important;} .down{color:#dc2626!important;}
.neu {color:#1d4ed8!important;} .warn{color:#d97706!important;}

/* 버전 배지 */
.ver-badge{display:inline-block;font-size:11px;padding:3px 10px;
  border-radius:20px;font-weight:600;background:#e0e7ff;color:#3730a3;}

/* 섹션 타이틀 */
.stitle{font-size:12px;font-weight:700;color:#475569;
  text-transform:uppercase;letter-spacing:1.5px;
  border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-bottom:14px;}

/* 인물카드 */
.pcard{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
  padding:16px 18px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.04);}

/* AI 거래 행 */
.ai-row{padding:8px 12px;border-radius:8px;margin-bottom:4px;
  font-size:13px;border:1px solid #e2e8f0;background:#fff;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  상수
# ══════════════════════════════════════════════════════════════════════════════
PORTFOLIO = {
    "김건희": {"color":"#2563eb","holdings":[
        {"ticker":"BULZ","shares":6842,"avg_cost":19.61}]},
    "김주열": {"color":"#16a34a","holdings":[
        {"ticker":"BULZ","shares":1532,"avg_cost":21.1935},
        {"ticker":"TQQQ","shares":14,  "avg_cost":43.01}]},
    "김주원": {"color":"#d97706","holdings":[
        {"ticker":"BULZ","shares":1140,"avg_cost":21.067},
        {"ticker":"TQQQ","shares":12,  "avg_cost":43.0450}]},
}
TC = {"BULZ":"#ea580c","TQQQ":"#2563eb"}

# ══════════════════════════════════════════════════════════════════════════════
#  클라우드 저장소 — GitHub Gist 기반
#  로컬 실행 시: json 파일 fallback 자동 사용
#  Streamlit Cloud 배포 시: secrets.toml 에 GIST_TOKEN, GIST_ID 설정
# ══════════════════════════════════════════════════════════════════════════════

# 로컬 fallback 파일명
_LOCAL_FILES = {
    "trades":         "trades.json",
    "ai_trades":      "ai_trades.json",
    "model_versions": "model_versions.json",
    "daily_signals":  "daily_signals.json",
}

def _is_cloud() -> bool:
    """Streamlit Cloud 환경 여부 확인"""
    try:
        token = st.secrets.get("GIST_TOKEN", "")
        return bool(token)
    except Exception:
        return False

def _gist_headers():
    token = st.secrets["GIST_TOKEN"]
    return {"Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"}

def _gist_pull() -> dict:
    """Gist에서 전체 데이터 pull"""
    try:
        gid = st.secrets["GIST_ID"]
        r = requests.get(f"https://api.github.com/gists/{gid}",
                         headers=_gist_headers(), timeout=10)
        if r.status_code == 200:
            files = r.json().get("files", {})
            result = {}
            for key in _LOCAL_FILES:
                fname = f"{key}.json"
                if fname in files:
                    content = files[fname].get("content", "{}")
                    try:
                        result[key] = json.loads(content)
                    except Exception:
                        result[key] = None
            return result
    except Exception:
        pass
    return {}

def _gist_push(key: str, data):
    """Gist에 특정 키 데이터 push"""
    try:
        gid = st.secrets["GIST_ID"]
        payload = {"files": {f"{key}.json": {"content": json.dumps(data, ensure_ascii=False, indent=2)}}}
        requests.patch(f"https://api.github.com/gists/{gid}",
                       headers=_gist_headers(),
                       json=payload, timeout=10)
    except Exception:
        pass

def _load_local(key: str, default):
    path = _LOCAL_FILES[key]
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def _save_local(key: str, data):
    path = _LOCAL_FILES[key]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _save(key: str, data):
    """환경에 따라 로컬 또는 Gist에 저장"""
    if _is_cloud():
        _gist_push(key, data)
    else:
        _save_local(key, data)

def load_all():
    """앱 시작 시 한 번만 실행 — 모든 데이터 로드"""
    if "data_loaded" in st.session_state:
        return

    defaults = {
        "trades":         {n: [] for n in PORTFOLIO},
        "ai_trades":      {"BULZ": [], "TQQQ": []},
        "model_versions": [],
        "daily_signals":  [],
    }

    if _is_cloud():
        # Gist에서 전체 pull (네트워크 1회)
        pulled = _gist_pull()
        for key, default in defaults.items():
            val = pulled.get(key)
            st.session_state[key] = val if val is not None else default
    else:
        # 로컬 파일에서 로드
        for key, default in defaults.items():
            st.session_state[key] = _load_local(key, default)

    st.session_state.data_loaded = True

load_all()

# ══════════════════════════════════════════════════════════════════════════════
#  데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def load_price(ticker, days=180):
    end=datetime.now(); start=end-timedelta(days=days)
    df=yf.download(ticker,start=start.strftime("%Y-%m-%d"),
                   end=end.strftime("%Y-%m-%d"),auto_adjust=True,progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]].dropna()

@st.cache_data(ttl=1800)
def load_range(ticker, start, end):
    df=yf.download(ticker,start=start,end=end,auto_adjust=True,progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index)
    return df[["Open","High","Low","Close","Volume"]].dropna()

# ══════════════════════════════════════════════════════════════════════════════
#  지표 계산
# ══════════════════════════════════════════════════════════════════════════════
def calc_ind(df, p):
    d=df.copy(); c=d["Close"]
    d["EMA_s"]=ta.trend.ema_indicator(c,window=p["ema_s"])
    d["EMA_l"]=ta.trend.ema_indicator(c,window=p["ema_l"])
    d["RSI"]  =ta.momentum.rsi(c,window=p["rsi_p"])
    macd=ta.trend.MACD(c,window_fast=p["mf"],window_slow=p["ms"],window_sign=p["mg"])
    d["MACD_h"]=macd.macd_diff()
    bb=ta.volatility.BollingerBands(c,window=20,window_dev=2)
    d["BB_u"]=bb.bollinger_hband(); d["BB_m"]=bb.bollinger_mavg()
    d["BB_l"]=bb.bollinger_lband()
    d["VOL_MA"]=d["Volume"].rolling(p.get("vp",20)).mean()
    if "atr_p" in p:
        d["ATR"]=ta.volatility.average_true_range(d["High"],d["Low"],c,window=p["atr_p"])
        d["ATR_MA"]=d["ATR"].rolling(p["atr_p"]).mean()
    return d.dropna()

# ══════════════════════════════════════════════════════════════════════════════
#  백테스트 엔진
# ══════════════════════════════════════════════════════════════════════════════
def run_bt(df, p, initial=100_000_000, ticker="TQQQ"):
    d=calc_ind(df,p); n=len(d)
    if n < p["ema_l"]+10: return {}
    cash=initial; shares=0.0; equity=[]; signals=[]; trades=[]
    in_pos=False; entry_p=0.0; entry_d=None
    # BULZ 분할진입용
    split_done=False; entry_p1=0.0; shares_1=0.0
    cash_res=0.0; peak=0.0
    comm=0.0015
    is_bulz = ticker=="BULZ"

    for i in range(1, n):
        price=float(d["Close"].iloc[i]); rv=d["RSI"].iloc[i]
        trend= d["EMA_s"].iloc[i]>d["EMA_l"].iloc[i]
        p_ok = price>d["EMA_s"].iloc[i]
        r_ok = p["rl"]<=rv<=p["rh"]
        m_ok = d["MACD_h"].iloc[i]>0 and d["MACD_h"].iloc[i-1]<=0
        v_ok = d["Volume"].iloc[i]>=d["VOL_MA"].iloc[i]
        atr_ok=True
        if "ATR" in d.columns and "ATR_MA" in d.columns:
            an=d["ATR"].iloc[i]; am=d["ATR_MA"].iloc[i]
            if pd.notna(an) and pd.notna(am) and am>0:
                atr_ok=an<=am*p.get("atr_m",2.5)
        buy_sig = trend and p_ok and r_ok and m_ok and v_ok and atr_ok
        bb_ok   = price>d["BB_m"].iloc[i]

        # ── 매수 ──────────────────────────────────────────────────────────
        if not in_pos and buy_sig:
            if is_bulz:  # 분할 진입
                invest1=cash*0.5; shares_1=invest1/(price*(1+comm))
                cash_res=cash*0.5; cash=0.0; shares=shares_1
                split_done=False; entry_p1=price; entry_p=price; peak=price
                signals.append((d.index[i],price,"BUY1"))
            else:
                shares=cash/(price*(1+comm)); cash=0.0
                entry_p=price; peak=price
                signals.append((d.index[i],price,"BUY"))
            in_pos=True; entry_d=d.index[i]

        # BULZ 2차 진입
        elif is_bulz and in_pos and not split_done and cash_res>0 \
             and rv<=p["rl"] and trend and atr_ok:
            sh2=cash_res/(price*(1+comm)); shares+=sh2
            entry_p=(shares_1*entry_p1+sh2*price)/shares
            cash_res=0.0; split_done=True
            signals.append((d.index[i],price,"BUY2"))

        # ── 매도 ──────────────────────────────────────────────────────────
        stop      = in_pos and price<=entry_p*(1-p["sl"])
        trail     = is_bulz and in_pos and peak>0 and price<=peak*(1-p.get("tr",0.20))
        ema_exit  = in_pos and not trend
        rsi_exit  = in_pos and rv>p["rh"]
        atr_exit  = in_pos and not atr_ok
        sell_sig  = stop or trail or ema_exit or rsi_exit or atr_exit

        if in_pos and sell_sig:
            proceeds=shares*price*(1-comm)
            ret=(price/entry_p-1)*100
            why=("손절" if stop else "트레일링" if trail else
                 "EMA이탈" if ema_exit else "RSI과매수" if rsi_exit else "ATR위험")
            trades.append({"매수일":entry_d.strftime("%Y-%m-%d"),
                "매도일":d.index[i].strftime("%Y-%m-%d"),
                "매수가":round(entry_p,2),"매도가":round(price,2),
                "수익률":round(ret,2),"보유일수":(d.index[i]-entry_d).days,
                "청산사유":why,"분할":"2차" if split_done else "1차",
                "결과":"✅" if ret>0 else "❌"})
            cash=proceeds+(cash_res if is_bulz else 0); cash_res=0.0
            shares=0.0; in_pos=False; split_done=False; peak=0.0
            signals.append((d.index[i],price,"SELL"))

        if in_pos and price>peak: peak=price
        equity.append(cash+(cash_res if is_bulz else 0)+shares*price)

    if in_pos:
        lp=float(d["Close"].iloc[-1])
        equity[-1]=cash+(cash_res if is_bulz else 0)+shares*lp

    eq=pd.Series(equity,index=d.index[1:])
    if len(eq)==0: return {}
    fe=eq.iloc[-1]; tr=(fe/initial-1)*100
    yrs=(d.index[-1]-d.index[0]).days/365.25
    cagr=((fe/initial)**(1/max(yrs,.01))-1)*100
    dr=eq.pct_change().dropna()
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    rm=eq.cummax(); dd=(eq-rm)/rm*100; mdd=dd.min()
    wins=[t for t in trades if t["수익률"]>0]
    losses=[t for t in trades if t["수익률"]<=0]
    wr=len(wins)/len(trades)*100 if trades else 0
    aw=np.mean([t["수익률"] for t in wins]) if wins else 0
    al=np.mean([t["수익률"] for t in losses]) if losses else 0
    bh=(float(d["Close"].iloc[-1])/float(d["Close"].iloc[0])-1)*100
    return {"eq":eq,"dd":dd,"signals":signals,"df":d,"trades":trades,
            "p":p,"ticker":ticker,
            "s":{"ret":round(tr,2),"cagr":round(cagr,2),"mdd":round(mdd,2),
                 "sharpe":round(sharpe,2),"trades":len(trades),
                 "wr":round(wr,2),"wins":len(wins),"losses":len(losses),
                 "aw":round(aw,2),"al":round(al,2),
                 "pf":round(abs(aw/al),2) if al!=0 else 99,
                 "bh":round(bh,2),"final":round(fe)}}

# ══════════════════════════════════════════════════════════════════════════════
#  모델 파라미터 기본값
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_TQQQ = dict(ema_s=20,ema_l=60,rsi_p=14,rl=40,rh=70,mf=12,ms=26,mg=9,sl=0.10,vp=20)
DEFAULT_BULZ = dict(ema_s=30,ema_l=90,rsi_p=21,rl=35,rh=75,mf=19,ms=39,mg=9,
                    sl=0.15,tr=0.20,atr_p=14,atr_m=2.5,vp=20)

def get_active_model(ticker):
    """버전 기록에서 가장 최신 모델 파라미터 반환"""
    versions = st.session_state.model_versions
    ticker_versions = [v for v in versions if v["ticker"]==ticker]
    if not ticker_versions:
        return DEFAULT_TQQQ if ticker=="TQQQ" else DEFAULT_BULZ, "v1.0"
    latest = sorted(ticker_versions, key=lambda x: x["version"])[-1]
    return latest["params"], latest["version"]

# ══════════════════════════════════════════════════════════════════════════════
#  AI 실시간 신호 생성 (오늘 날짜 기준)
# ══════════════════════════════════════════════════════════════════════════════
def gen_signal(ticker, df, p):
    if df.empty or len(df)<60:
        return {"sig":"HOLD","score":0,"strength":0,"rsi":0,
                "trend":False,"price":0,"reasons":[],"detail":{},
                "bb_u":0,"bb_l":0,"bb_m":0}
    d=calc_ind(df,p); c=df["Close"]
    price=float(c.iloc[-1])
    rv=float(d["RSI"].iloc[-1])
    es=float(d["EMA_s"].iloc[-1]); el=float(d["EMA_l"].iloc[-1])
    mh=float(d["MACD_h"].iloc[-1]); mh_p=float(d["MACD_h"].iloc[-2])
    vm=float(d["VOL_MA"].iloc[-1]); vol=float(df["Volume"].iloc[-1])
    bb_u=float(d["BB_u"].iloc[-1]); bb_l=float(d["BB_l"].iloc[-1])
    bb_m=float(d["BB_m"].iloc[-1])
    score=0; reasons=[]
    checks = {}

    # 추세
    if es>el:   score+=2; reasons.append("✅ EMA 정배열 — 상승 추세 확인"); checks["EMA추세"]="정배열↑"
    else:       score-=2; reasons.append("❌ EMA 역배열 — 하락 추세"); checks["EMA추세"]="역배열↓"
    if price>es: score+=1; reasons.append("✅ 가격이 EMA 위 — 강세 포지션"); checks["가격위치"]="EMA위"
    else:        score-=1; reasons.append("❌ 가격이 EMA 아래 — 약세"); checks["가격위치"]="EMA아래"

    # 모멘텀
    if p["rl"]<=rv<=p["rh"]:
        score+=2; reasons.append(f"✅ RSI {rv:.1f} — 진입 적정 구간({p['rl']}~{p['rh']})"); checks["RSI"]=f"{rv:.1f} 적정"
    elif rv<p["rl"]:
        score+=1; reasons.append(f"⚠️ RSI {rv:.1f} — 과매도, 반등 기대 가능"); checks["RSI"]=f"{rv:.1f} 과매도"
    else:
        score-=2; reasons.append(f"❌ RSI {rv:.1f} — 과매수, 청산 고려"); checks["RSI"]=f"{rv:.1f} 과매수"

    if mh>0 and mh_p<=0:   score+=3; reasons.append("🔥 MACD 히스토그램 음→양 전환 (강력 매수 신호)"); checks["MACD"]="양전환🔥"
    elif mh>0:              score+=1; reasons.append("✅ MACD 히스토그램 양수 유지"); checks["MACD"]="양수유지"
    elif mh<0 and mh_p>=0: score-=2; reasons.append("❌ MACD 히스토그램 양→음 전환 (매도 신호)"); checks["MACD"]="음전환❌"
    else:                   score-=1; reasons.append("❌ MACD 히스토그램 음수"); checks["MACD"]="음수"

    # 거래량
    if vol>=vm: score+=1; reasons.append("✅ 거래량 20일 평균 이상 — 시장 참여 확인"); checks["거래량"]="평균이상"
    else:       reasons.append("⚠️ 거래량 평균 미달 — 신호 신뢰도 낮음"); checks["거래량"]="미달"

    # 볼린저
    if price<bb_l:   score+=1; reasons.append("✅ 볼린저 하단 근접 — 반등 가능성"); checks["BB"]="하단근접"
    elif price>bb_u: score-=1; reasons.append("⚠️ 볼린저 상단 돌파 — 과열 주의"); checks["BB"]="상단돌파"
    else:            reasons.append(f"✅ 볼린저 밴드 내 정상 범위"); checks["BB"]="정상범위"

    # ATR (BULZ)
    if "ATR" in d.columns and "ATR_MA" in d.columns:
        an=float(d["ATR"].iloc[-1]); am=float(d["ATR_MA"].iloc[-1])
        if pd.notna(an) and pd.notna(am) and am>0:
            ratio=an/am
            if ratio>p.get("atr_m",2.5):
                score-=2; reasons.append(f"❌ ATR {ratio:.1f}배 — 변동성 폭발, 진입 위험"); checks["ATR"]=f"{ratio:.1f}배 위험"
            else:
                reasons.append(f"✅ ATR 정상 범위 ({ratio:.1f}배)"); checks["ATR"]=f"{ratio:.1f}배 정상"

    sig = "BUY" if score>=5 else "SELL" if score<=-2 else "HOLD"
    strength=min(max((score+4)/10*100,0),100)
    return {"sig":sig,"score":score,"strength":round(strength),
            "rsi":round(rv,1),"trend":es>el,"price":round(price,2),
            "reasons":reasons,"detail":checks,
            "bb_u":round(bb_u,2),"bb_l":round(bb_l,2),"bb_m":round(bb_m,2),
            "macd_h":round(mh,4)}

# ══════════════════════════════════════════════════════════════════════════════
#  AI 자동 거래 기록 & 신호 저장
# ══════════════════════════════════════════════════════════════════════════════
def record_ai_signal(ticker, sig, model_ver):
    today=datetime.now().strftime("%Y-%m-%d")
    daily=st.session_state.daily_signals

    # 오늘 이미 기록됐으면 업데이트
    existing=[i for i,s in enumerate(daily)
              if s["date"]==today and s["ticker"]==ticker]
    entry={"date":today,"ticker":ticker,"signal":sig["sig"],
           "price":sig["price"],"score":sig["score"],
           "strength":sig["strength"],"rsi":sig["rsi"],
           "trend":"정배열" if sig["trend"] else "역배열",
           "model_ver":model_ver,
           "reasons":sig["reasons"]}
    if existing:
        daily[existing[0]]=entry
    else:
        daily.append(entry)
    _save("daily_signals", daily)

    # AI 거래 기록 (BUY/SELL일 때만)
    ai_trades=st.session_state.ai_trades
    if ticker not in ai_trades: ai_trades[ticker]=[]

    # 현재 포지션 상태 확인
    tl=ai_trades[ticker]
    in_pos=any(t.get("status")=="open" for t in tl)

    if sig["sig"]=="BUY" and not in_pos:
        tl.append({"date":today,"type":"BUY","ticker":ticker,
            "price":sig["price"],"status":"open","model_ver":model_ver,
            "reason":sig["reasons"][:3],"score":sig["score"]})
        _save("ai_trades", ai_trades)
        return "BUY_RECORDED"

    elif sig["sig"]=="SELL" and in_pos:
        # 매도 처리
        for t in tl:
            if t.get("status")=="open":
                ret=round((sig["price"]/t["price"]-1)*100,2)
                t["sell_date"]=today; t["sell_price"]=sig["price"]
                t["status"]="closed"; t["ret"]=ret
                t["sell_reason"]=sig["reasons"][:3]
                t["result"]="✅ 수익" if ret>0 else "❌ 손실"
        _save("ai_trades", ai_trades)
        return "SELL_RECORDED"
    return "NO_ACTION"

# ══════════════════════════════════════════════════════════════════════════════
#  모델 자동 업데이트 (매일 재백테스트 → 최적 파라미터 탐색)
# ══════════════════════════════════════════════════════════════════════════════
def auto_update_model(ticker, df):
    """최근 누적 데이터로 파라미터 최적화 → 새 버전 저장"""
    versions=st.session_state.model_versions
    ticker_vers=[v for v in versions if v["ticker"]==ticker]
    ver_num=len(ticker_vers)+1
    ver_str=f"v{ver_num}.{datetime.now().strftime('%m%d')}"

    best_score=-999; best_p=None; best_s=None

    # 파라미터 탐색 범위
    if ticker=="TQQQ":
        search=[
            dict(ema_s=s,ema_l=l,rsi_p=r,rl=rl,rh=rh,mf=mf,ms=ms,mg=9,sl=sl,vp=20)
            for s in [15,20,25] for l in [50,60,80]
            for r in [10,14] for rl in [35,40] for rh in [65,70]
            for mf in [10,12] for ms in [24,26]
            for sl in [0.08,0.10,0.12]
        ][:30]  # 최대 30개 조합
    else:
        search=[
            dict(ema_s=s,ema_l=l,rsi_p=r,rl=rl,rh=rh,mf=mf,ms=ms,mg=9,
                 sl=sl,tr=tr,atr_p=14,atr_m=atr_m,vp=20)
            for s in [25,30,35] for l in [80,90,100]
            for r in [18,21] for rl in [30,35] for rh in [72,75]
            for mf in [17,19] for ms in [36,39]
            for sl in [0.12,0.15] for tr in [0.18,0.20]
            for atr_m in [2.0,2.5]
        ][:30]

    progress=st.progress(0, text=f"{ticker} 모델 최적화 중...")
    for i,p in enumerate(search):
        try:
            res=run_bt(df,p,ticker=ticker)
            if res:
                s=res["s"]
                # 종합 점수: 샤프 40% + 승률 30% + 수익률 30%
                score=(min(max(s["sharpe"],0),3)/3*40
                      +min(s["wr"],100)/100*30
                      +min(max(s["ret"],0),300)/300*30)
                if score>best_score:
                    best_score=score; best_p=p; best_s=s
        except: pass
        progress.progress((i+1)/len(search),
                          text=f"{ticker} 최적화 중... {i+1}/{len(search)}")
    progress.empty()

    if best_p is None:
        return None, "최적화 실패"

    new_ver={"version":ver_str,"ticker":ticker,"date":datetime.now().strftime("%Y-%m-%d"),
             "params":best_p,"stats":best_s,"score":round(best_score,1),
             "data_days":len(df),
             "note":f"자동 업데이트 — 데이터 {len(df)}일 기반"}
    versions.append(new_ver)
    _save("model_versions", versions)
    return new_ver, ver_str

# ══════════════════════════════════════════════════════════════════════════════
#  수동 거래 성과 계산
# ══════════════════════════════════════════════════════════════════════════════
def manual_stats(person):
    trades=st.session_state.trades.get(person,[])
    completed=[]; bq={}
    for t in sorted(trades,key=lambda x:x["date"]):
        tk=t["ticker"]
        if tk not in bq: bq[tk]=[]
        if t["type"]=="BUY":
            bq[tk].append({"date":t["date"],"price":t["price"],"qty":t["qty"]})
        elif t["type"]=="SELL" and bq.get(tk):
            sq=t["qty"]; sp=t["price"]
            while sq>0 and bq[tk]:
                b=bq[tk][0]; m=min(b["qty"],sq)
                completed.append({"ticker":tk,"buy_date":b["date"],
                    "sell_date":t["date"],"buy_price":b["price"],
                    "sell_price":sp,"qty":m,
                    "pnl":round((sp/b["price"]-1)*100,2)})
                b["qty"]-=m; sq-=m
                if b["qty"]==0: bq[tk].pop(0)
    if not completed:
        return {"n":0,"wins":0,"losses":0,"wr":0,"aw":0,"al":0,"total":0,"list":[]}
    wins=[c for c in completed if c["pnl"]>0]
    losses=[c for c in completed if c["pnl"]<=0]
    return {"n":len(completed),"wins":len(wins),"losses":len(losses),
            "wr":round(len(wins)/len(completed)*100,1),
            "aw":round(np.mean([c["pnl"] for c in wins]),2) if wins else 0,
            "al":round(np.mean([c["pnl"] for c in losses]),2) if losses else 0,
            "total":round(sum(c["pnl"] for c in completed),2),"list":completed}

def ai_stats(ticker):
    tl=st.session_state.ai_trades.get(ticker,[])
    closed=[t for t in tl if t.get("status")=="closed"]
    if not closed:
        return {"n":0,"wins":0,"wr":0,"aw":0,"al":0,"total":0}
    wins=[t for t in closed if t.get("ret",0)>0]
    losses=[t for t in closed if t.get("ret",0)<=0]
    return {"n":len(closed),"wins":len(wins),
            "wr":round(len(wins)/len(closed)*100,1) if closed else 0,
            "aw":round(np.mean([t["ret"] for t in wins]),2) if wins else 0,
            "al":round(np.mean([t["ret"] for t in losses]),2) if losses else 0,
            "total":round(sum(t.get("ret",0) for t in closed),2)}

# ══════════════════════════════════════════════════════════════════════════════
#  차트
# ══════════════════════════════════════════════════════════════════════════════
WL=dict(template="plotly_white",paper_bgcolor="#fff",plot_bgcolor="#fafafa",
        font=dict(color="#374151",size=11),margin=dict(l=10,r=10,t=40,b=10))

def chart_price(ticker, df, sig, ai_tl):
    if df.empty: return go.Figure()
    color=TC[ticker]; fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="종가",
        line=dict(color=color,width=2),
        hovertemplate="%{x|%m/%d}  $%{y:,.2f}<extra></extra>"))

    # AI 매수/매도 마커
    ai_buys =[(pd.Timestamp(t["date"]),t["price"]) for t in ai_tl
               if t.get("type")=="BUY" and t.get("price")]
    ai_sells=[(pd.Timestamp(t["sell_date"]),t["sell_price"]) for t in ai_tl
               if t.get("status")=="closed" and t.get("sell_price")]
    if ai_buys:
        bx,by=zip(*ai_buys)
        fig.add_trace(go.Scatter(x=list(bx),y=[p*0.96 for p in by],
            mode="markers",name="AI 매수",
            marker=dict(symbol="triangle-up",size=14,color="#16a34a"),
            hovertemplate="AI매수 $%{y:,.2f}<extra></extra>"))
    if ai_sells:
        sx,sy=zip(*ai_sells)
        fig.add_trace(go.Scatter(x=list(sx),y=[p*1.04 for p in sy],
            mode="markers",name="AI 매도",
            marker=dict(symbol="triangle-down",size=14,color="#dc2626"),
            hovertemplate="AI매도 $%{y:,.2f}<extra></extra>"))

    # 볼린저밴드
    if sig and sig["bb_u"]:
        fig.add_hline(y=sig["bb_u"],line_dash="dot",
                      line_color="#94a3b8",line_width=1,
                      annotation_text="BB상단")
        fig.add_hline(y=sig["bb_l"],line_dash="dot",
                      line_color="#94a3b8",line_width=1,
                      annotation_text="BB하단")

    fig.update_layout(**WL,title=f"{ticker} 가격 추이 + AI 매매",
                      height=320,hovermode="x unified",
                      legend=dict(orientation="h",y=1.08))
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0",tickprefix="$")
    return fig

def chart_pv_compare(price_data):
    fig=go.Figure()
    for person,info in PORTFOLIO.items():
        sl=[]
        for h in info["holdings"]:
            tk=h["ticker"]; qty=h["shares"]
            if tk in price_data and not price_data[tk].empty:
                sl.append(price_data[tk]["Close"]*qty)
        if not sl: continue
        s=pd.concat(sl,axis=1).sum(axis=1).dropna()
        norm=s/s.iloc[0]*100
        fig.add_trace(go.Scatter(x=norm.index,y=norm.round(2),
            name=person,line=dict(color=info["color"],width=2.2),
            hovertemplate=f"{person}  %{{x|%m/%d}}  %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**WL,title="3인 포트폴리오 수익률 비교 (기준=100)",
                      height=360,hovermode="x unified",
                      legend=dict(orientation="h",y=1.08))
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

def chart_bt(res):
    if not res: return go.Figure()
    ticker=res["ticker"]; color=TC[ticker]
    eq=res["eq"]; dd=res["dd"]; d=res["df"]; sigs=res["signals"]
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        row_heights=[0.52,0.28,0.20],vertical_spacing=0.03,
        subplot_titles=[f"{ticker} 가격+신호","포트폴리오 가치","낙폭(%)"])
    fig.add_trace(go.Scatter(x=d.index,y=d["Close"],name="종가",
        line=dict(color=color,width=1.8)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d["EMA_s"],
        line=dict(color="#94a3b8",width=1,dash="dot"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d["EMA_l"],
        line=dict(color="#cbd5e1",width=1,dash="dot"),showlegend=False),row=1,col=1)
    buys1=[(dt,p) for dt,p,t in sigs if t in("BUY","BUY1")]
    buys2=[(dt,p) for dt,p,t in sigs if t=="BUY2"]
    sells=[(dt,p) for dt,p,t in sigs if t=="SELL"]
    if buys1:
        bx,by=zip(*buys1)
        fig.add_trace(go.Scatter(x=list(bx),y=[p*0.96 for p in by],
            mode="markers",name="매수1",
            marker=dict(symbol="triangle-up",size=10,color="#16a34a")),row=1,col=1)
    if buys2:
        bx2,by2=zip(*buys2)
        fig.add_trace(go.Scatter(x=list(bx2),y=[p*0.93 for p in by2],
            mode="markers",name="매수2",
            marker=dict(symbol="triangle-up",size=8,color="#86efac")),row=1,col=1)
    if sells:
        sx,sy=zip(*sells)
        fig.add_trace(go.Scatter(x=list(sx),y=[p*1.04 for p in sy],
            mode="markers",name="매도",
            marker=dict(symbol="triangle-down",size=10,color="#dc2626")),row=1,col=1)
    r=_hexrgb(color)
    fig.add_trace(go.Scatter(x=eq.index,y=eq,name="포트폴리오",
        line=dict(color=color,width=2),
        fill="tozeroy",fillcolor=f"rgba({r},.1)"),row=2,col=1)
    fig.add_trace(go.Scatter(x=dd.index,y=dd,name="낙폭",
        line=dict(color="#dc2626",width=1),
        fill="tozeroy",fillcolor="rgba(220,38,38,.1)"),row=3,col=1)
    fig.update_layout(template="plotly_white",paper_bgcolor="#fff",
        plot_bgcolor="#fafafa",font=dict(color="#374151",size=11),
        height=600,hovermode="x unified",
        legend=dict(orientation="h",y=1.02,font_size=10),
        margin=dict(l=10,r=10,t=40,b=10))
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

def chart_version_perf():
    versions=st.session_state.model_versions
    if not versions: return go.Figure()
    fig=go.Figure()
    for tk,color in [("TQQQ","#2563eb"),("BULZ","#ea580c")]:
        tvs=[v for v in versions if v["ticker"]==tk]
        if not tvs: continue
        fig.add_trace(go.Scatter(
            x=[v["version"] for v in tvs],
            y=[v["stats"]["ret"] for v in tvs],
            mode="lines+markers+text",name=tk,
            line=dict(color=color,width=2),
            marker=dict(size=10),
            text=[f"{v['stats']['ret']:+.1f}%" for v in tvs],
            textposition="top center"))
    fig.update_layout(**WL,title="모델 버전별 수익률 추이",height=300,
                      legend=dict(orientation="h",y=1.08))
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0",ticksuffix="%")
    return fig

def _hexrgb(h):
    h=h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

def kcard(lbl,val,css="",sub=""):
    sh=f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="kcard"><div class="lbl">{lbl}</div>'
            f'<div class="val {css}">{val}</div>{sh}</div>')

# ══════════════════════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
with st.spinner("시세 로드 중..."):
    df_b=load_price("BULZ",days=180); df_t=load_price("TQQQ",days=180)
    price_data={"BULZ":df_b,"TQQQ":df_t}
    bp=float(df_b["Close"].iloc[-1]) if not df_b.empty else 0.0
    tp=float(df_t["Close"].iloc[-1]) if not df_t.empty else 0.0
    prices={"BULZ":bp,"TQQQ":tp}

# 활성 모델 파라미터
bulz_p,bulz_ver=get_active_model("BULZ")
tqqq_p,tqqq_ver=get_active_model("TQQQ")

# 신호 생성
b_sig=gen_signal("BULZ",df_b,bulz_p)
t_sig=gen_signal("TQQQ",df_t,tqqq_p)

# 오늘 신호 자동 기록
record_ai_signal("BULZ",b_sig,bulz_ver)
record_ai_signal("TQQQ",t_sig,tqqq_ver)

today=datetime.now().strftime("%Y년 %m월 %d일")

# ── 헤더 ──────────────────────────────────────────────────────────────────────
col_h1,col_h2=st.columns([3,1])
with col_h1:
    st.title("📈 필승매매 시스템")
    st.caption(f"TQQQ · BULZ  |  {today}  |  "
               f"TQQQ {tqqq_ver} · BULZ {bulz_ver}")
with col_h2:
    st.markdown(f'<div style="text-align:right;padding-top:16px;">'
                f'<span class="ver-badge">TQQQ {tqqq_ver}</span>&nbsp;'
                f'<span class="ver-badge"style="background:#fef3c7;color:#92400e;">'
                f'BULZ {bulz_ver}</span></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  🚨 AI 매매 신호 (최상단)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="stitle">🤖 오늘의 AI 매매 신호</div>',
            unsafe_allow_html=True)
sc1,sc2=st.columns(2)
for col,(tk,sig,ver) in zip([sc1,sc2],[("BULZ",b_sig,bulz_ver),("TQQQ",t_sig,tqqq_ver)]):
    with col:
        stype=sig["sig"]
        cls = "sig-buy" if stype=="BUY" else "sig-sell" if stype=="SELL" else "sig-hold"
        icon= "🟢 매수 신호" if stype=="BUY" else "🔴 매도 신호" if stype=="SELL" else "🔵 보유 유지"
        css = "up" if stype=="BUY" else "down" if stype=="SELL" else "neu"
        st.markdown(
            f'<div class="sig {cls}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:17px;font-weight:700;color:{TC[tk]};">{tk}</span>'
            f'<span style="font-size:11px;background:#f1f5f9;color:#475569;'
            f'padding:2px 8px;border-radius:10px;">{ver}</span></div>'
            f'<div style="font-size:28px;font-weight:800;margin:6px 0;" class="{css}">'
            f'${sig["price"]:,.2f}</div>'
            f'<div style="font-size:14px;font-weight:600;" class="{css}">{icon}</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:6px;">'
            f'강도 {sig["strength"]}%  ·  RSI {sig["rsi"]}  ·  '
            f'EMA {"정배열↑" if sig["trend"] else "역배열↓"}</div>'
            f'</div>',unsafe_allow_html=True)

        with st.expander(f"📋 {tk} 신호 근거 상세"):
            for r in sig["reasons"]:
                st.write(r)
            st.markdown("---")
            st.markdown("**지표 요약**")
            for k,v in sig["detail"].items():
                st.write(f"• **{k}**: {v}")
            if stype=="BUY":
                st.success(f"💡 **{tk} 매수 추천**\n\n"
                           f"현재가 **${sig['price']:,.2f}**에 진입 고려\n\n"
                           f"모델 기준 손절가: **${sig['price']*(1-bulz_p.get('sl',0.15)):.2f}**"
                           if tk=="BULZ" else
                           f"현재가 **${sig['price']:,.2f}**에 진입 고려\n\n"
                           f"모델 기준 손절가: **${sig['price']*(1-tqqq_p.get('sl',0.10)):.2f}**")
            elif stype=="SELL":
                st.error(f"⚠️ **{tk} 매도 추천**\n\n"
                         f"현재가 **${sig['price']:,.2f}** 청산 검토")
            else:
                st.info(f"📌 **{tk} 보유 유지** — 명확한 진입·청산 신호 없음")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    c1,c2=st.columns(2)
    with c1: sd=st.date_input("백테스트 시작",value=datetime.now()-timedelta(days=365*3))
    with c2: ed=st.date_input("종료",value=datetime.now())
    init_cash=st.number_input("초기자금(원)",value=100_000_000,step=10_000_000,format="%d")
    st.markdown("---")
    run_bt_btn=st.button("▶ 백테스트 실행",type="primary",use_container_width=True)
    st.markdown("---")
    st.markdown("### 🔄 모델 자동 업데이트")
    st.caption("최근 누적 데이터로 재최적화 → 새 버전 저장")
    upd_tqqq=st.button("TQQQ 모델 업데이트",use_container_width=True)
    upd_bulz=st.button("BULZ 모델 업데이트",use_container_width=True)
    if upd_tqqq:
        with st.spinner("TQQQ 최적화 중..."):
            df_t_full=load_range("TQQQ",str(sd),str(ed))
            nv,ver=auto_update_model("TQQQ",df_t_full)
        if nv:
            st.success(f"TQQQ {ver} 업데이트 완료!\n수익률 {nv['stats']['ret']:+.1f}%")
            st.rerun()
    if upd_bulz:
        with st.spinner("BULZ 최적화 중..."):
            df_b_full=load_range("BULZ",str(sd),str(ed))
            nv,ver=auto_update_model("BULZ",df_b_full)
        if nv:
            st.success(f"BULZ {ver} 업데이트 완료!\n수익률 {nv['stats']['ret']:+.1f}%")
            st.rerun()

# 백테스트 실행
if run_bt_btn or "bt_t" not in st.session_state:
    with st.spinner("백테스트 실행 중..."):
        df_t_r=load_range("TQQQ",str(sd),str(ed))
        df_b_r=load_range("BULZ",str(sd),str(ed))
        st.session_state.bt_t=run_bt(df_t_r,tqqq_p,float(init_cash),"TQQQ") if not df_t_r.empty else {}
        st.session_state.bt_b=run_bt(df_b_r,bulz_p,float(init_cash),"BULZ") if not df_b_r.empty else {}

bt_t=st.session_state.get("bt_t",{})
bt_b=st.session_state.get("bt_b",{})

# ══════════════════════════════════════════════════════════════════════════════
#  탭
# ══════════════════════════════════════════════════════════════════════════════
tabs=st.tabs(["📊 포트폴리오","🤖 AI 매매일지",
              "📝 내 거래 입력","⚔️ AI vs 나 비교",
              "🔬 백테스트","📈 모델 버전 관리"])

# ── 탭1: 포트폴리오 현황 ──────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="stitle">전체 포트폴리오 현황</div>',
                unsafe_allow_html=True)
    pc1,pc2=st.columns(2)
    for col,tk,pr,sig in[(pc1,"BULZ",bp,b_sig),(pc2,"TQQQ",tp,t_sig)]:
        with col:
            css="up" if sig["sig"]=="BUY" else "down" if sig["sig"]=="SELL" else "neu"
            st.markdown(kcard(f"{tk} 현재가",f"${pr:,.2f}",css,
                f"신호:{sig['sig']} · RSI {sig['rsi']} · 강도 {sig['strength']}%"),
                unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    for person,info in PORTFOLIO.items():
        pv_total=0; pv_cost=0; details=[]
        for h in info["holdings"]:
            tk=h["ticker"]; val=h["shares"]*prices.get(tk,0)
            cost=h["shares"]*h["avg_cost"]; pnl=val-cost
            pnl_pct=(prices.get(tk,0)/h["avg_cost"]-1)*100
            pv_total+=val; pv_cost+=cost
            details.append((tk,h["shares"],h["avg_cost"],prices.get(tk,0),val,pnl_pct))
        total_pnl=pv_total-pv_cost
        total_pct=(pv_total/pv_cost-1)*100 if pv_cost>0 else 0
        color=info["color"]
        st.markdown(
            f'<div class="pcard" style="border-left:4px solid {color};">'
            f'<span style="font-size:15px;font-weight:700;color:{color};">'
            f'{person}</span>',unsafe_allow_html=True)
        cols=st.columns(4)
        cols[0].markdown(kcard("총 평가액",f"${pv_total:,.0f}",
            "up" if total_pnl>=0 else "down"),unsafe_allow_html=True)
        cols[1].markdown(kcard("총 손익",f"${total_pnl:+,.0f}",
            "up" if total_pnl>=0 else "down",f"{total_pct:+.2f}%"),
            unsafe_allow_html=True)
        for ci,(tk,qty,avg,price,val,pct) in enumerate(details):
            cols[ci+2].markdown(kcard(f"{tk} {qty:,}주",f"${val:,.0f}",
                "up" if pct>=0 else "down",
                f"단가${avg:.2f}→${price:.2f} ({pct:+.1f}%)"),
                unsafe_allow_html=True)
        st.markdown("</div><br>",unsafe_allow_html=True)
    st.plotly_chart(chart_pv_compare(price_data),use_container_width=True)

    # 종목 가격 차트
    c1,c2=st.columns(2)
    with c1:
        ai_tl_b=st.session_state.ai_trades.get("BULZ",[])
        st.plotly_chart(chart_price("BULZ",df_b,b_sig,ai_tl_b),
                        use_container_width=True)
    with c2:
        ai_tl_t=st.session_state.ai_trades.get("TQQQ",[])
        st.plotly_chart(chart_price("TQQQ",df_t,t_sig,ai_tl_t),
                        use_container_width=True)

# ── 탭2: AI 매매일지 ──────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="stitle">🤖 AI 자동매매 일지</div>',
                unsafe_allow_html=True)
    st.caption("AI 모델이 신호를 발생시킬 때마다 이유와 함께 자동 기록됩니다.")

    for tk,color in [("BULZ","#ea580c"),("TQQQ","#2563eb")]:
        st.markdown(f'<span style="color:{color};font-size:15px;font-weight:700;">'
                    f'{tk}</span>',unsafe_allow_html=True)
        ai_tl=st.session_state.ai_trades.get(tk,[])

        # 현재 오픈 포지션
        open_pos=[t for t in ai_tl if t.get("status")=="open"]
        if open_pos:
            op=open_pos[-1]
            cur_price=prices.get(tk,0)
            cur_ret=(cur_price/op["price"]-1)*100 if op["price"]>0 else 0
            st.markdown(
                f'<div class="sig sig-buy" style="margin-bottom:10px;">'
                f'<b>현재 보유 중</b>  |  매수가 ${op["price"]}  |  '
                f'현재가 ${cur_price:.2f}  |  '
                f'<span class="{"up" if cur_ret>=0 else "down"}">'
                f'미실현 {cur_ret:+.1f}%</span>  |  모델 {op["model_ver"]}'
                f'</div>',unsafe_allow_html=True)

        # 완료된 거래
        closed=[t for t in ai_tl if t.get("status")=="closed"]
        if closed:
            st.markdown(f"**완료된 거래 ({len(closed)}건)**")
            for t in reversed(closed[-10:]):
                ret=t.get("ret",0)
                css="up" if ret>=0 else "down"
                reasons_str=" / ".join(t.get("reason",[""])[:2])
                st.markdown(
                    f'<div class="ai-row">'
                    f'<b style="color:{color};">{t["date"]}</b> 매수 ${t["price"]}  →  '
                    f'<b>{t.get("sell_date","")} 매도 ${t.get("sell_price","")}</b>  |  '
                    f'<span class="{css}"><b>{ret:+.1f}%</b></span>  |  '
                    f'{t.get("result","")}  |  {t.get("sell_reason",[""])[0] if isinstance(t.get("sell_reason"),list) else t.get("sell_reason","")}'
                    f'</div>',unsafe_allow_html=True)
        else:
            st.info(f"{tk} 완료 거래 없음 (신호 발생 시 자동 기록됩니다)")

        ais=ai_stats(tk)
        if ais["n"]>0:
            a1,a2,a3,a4=st.columns(4)
            a1.markdown(kcard("AI 승률",f"{ais['wr']}%","up" if ais["wr"]>=50 else "down"),unsafe_allow_html=True)
            a2.markdown(kcard("평균 수익",f"+{ais['aw']}%","up"),unsafe_allow_html=True)
            a3.markdown(kcard("평균 손실",f"{ais['al']}%","down"),unsafe_allow_html=True)
            a4.markdown(kcard("총 거래",f"{ais['n']}건"),unsafe_allow_html=True)

        # 일별 신호 기록
        with st.expander(f"{tk} 일별 신호 기록"):
            ds=[s for s in st.session_state.daily_signals if s["ticker"]==tk]
            if ds:
                st.dataframe(pd.DataFrame(ds)[
                    ["date","signal","price","score","strength","rsi","trend","model_ver"]
                ].sort_values("date",ascending=False),
                use_container_width=True,hide_index=True,height=240)
        st.markdown("---")

# ── 탭3: 내 거래 입력 ─────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="stitle">📝 내 거래 직접 입력</div>',
                unsafe_allow_html=True)

    # 현재 AI 신호 요약 (참고용)
    st.markdown("**📌 AI 신호 참고 (오늘)**")
    ref1,ref2=st.columns(2)
    for col,(tk,sig) in zip([ref1,ref2],[("BULZ",b_sig),("TQQQ",t_sig)]):
        with col:
            css="up" if sig["sig"]=="BUY" else "down" if sig["sig"]=="SELL" else "neu"
            st.markdown(kcard(f"{tk} AI신호",sig["sig"],css,
                f"${sig['price']:,.2f} · RSI {sig['rsi']} · 강도 {sig['strength']}%"),
                unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    with st.form("trade_form",clear_on_submit=True):
        f1,f2,f3=st.columns(3)
        with f1:
            sel_p=st.selectbox("투자자",list(PORTFOLIO.keys()))
            sel_tk=st.selectbox("종목",["BULZ","TQQQ"])
            sel_tp=st.selectbox("유형",["BUY","SELL"])
        with f2:
            sel_dt=st.date_input("거래일",value=datetime.now())
            sel_pr=st.number_input("거래가 ($)",min_value=0.01,
                value=float(prices.get("BULZ",20.0)),step=0.01,format="%.4f")
        with f3:
            sel_qt=st.number_input("수량 (주)",min_value=1,value=100,step=1)
            sel_memo=st.text_input("메모",placeholder="예: AI 신호 따라 진입")
            sel_follow=st.checkbox("AI 신호 따름",value=True)
        submitted=st.form_submit_button("✅ 거래 추가",type="primary",
                                        use_container_width=True)
        if submitted:
            nt={"date":str(sel_dt),"ticker":sel_tk,"type":sel_tp,
                "price":round(sel_pr,4),"qty":int(sel_qt),"memo":sel_memo,
                "amount":round(sel_pr*sel_qt,2),
                "ai_signal":b_sig["sig"] if sel_tk=="BULZ" else t_sig["sig"],
                "followed_ai":sel_follow}
            if sel_p not in st.session_state.trades:
                st.session_state.trades[sel_p]=[]
            st.session_state.trades[sel_p].append(nt)
            _save("trades", st.session_state.trades)
            st.success(f"✅ {sel_p}  {sel_tp} {sel_tk} {sel_qt:,}주 @ ${sel_pr:.4f}")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 내 거래 내역 확인 & 수정")
    for person in PORTFOLIO:
        tl=st.session_state.trades.get(person,[])
        if not tl: continue
        color=PORTFOLIO[person]["color"]
        st.markdown(f'<span style="color:{color};font-weight:700;">{person}</span>',
                    unsafe_allow_html=True)
        edited=st.data_editor(pd.DataFrame(tl),use_container_width=True,
            hide_index=True,num_rows="dynamic",key=f"ed_{person}",
            height=min(38*len(tl)+38,260))
        cs,_=st.columns([1,5])
        with cs:
            if st.button(f"💾 저장",key=f"sv_{person}"):
                st.session_state.trades[person]=edited.to_dict("records")
                _save("trades", st.session_state.trades)
                st.success("저장 완료!"); st.rerun()

# ── 탭4: AI vs 나 비교 ───────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="stitle">⚔️ AI 모델 vs 내 실거래 성과 비교</div>',
                unsafe_allow_html=True)
    for tk,color in [("BULZ","#ea580c"),("TQQQ","#2563eb")]:
        st.markdown(f'<span style="color:{color};font-size:15px;font-weight:700;">{tk}</span>',
                    unsafe_allow_html=True)
        ais=ai_stats(tk)
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<span style="background:#e0e7ff;color:#3730a3;padding:2px 9px;'
                        'border-radius:20px;font-size:11px;font-weight:600;">🤖 AI 모델</span>',
                        unsafe_allow_html=True)
            if ais["n"]>0:
                a1,a2,a3=st.columns(3)
                a1.markdown(kcard("승률",f"{ais['wr']}%","up" if ais["wr"]>=50 else "down"),unsafe_allow_html=True)
                a2.markdown(kcard("평균 수익",f"+{ais['aw']}%","up"),unsafe_allow_html=True)
                a3.markdown(kcard("총 손익",f"{ais['total']:+.1f}%","up" if ais["total"]>=0 else "down"),unsafe_allow_html=True)
            else:
                st.info("AI 완료 거래 없음 (신호 기반 매매 기록 누적 중)")
        with c2:
            st.markdown('<span style="background:#fef9c3;color:#854d0e;padding:2px 9px;'
                        'border-radius:20px;font-size:11px;font-weight:600;">👤 내 실거래</span>',
                        unsafe_allow_html=True)
            # 해당 종목 보유자만 집계
            holders=[p for p,info in PORTFOLIO.items()
                     if any(h["ticker"]==tk for h in info["holdings"])]
            all_manual=[]
            for person in holders:
                ms=manual_stats(person)
                all_manual.extend(ms["list"])
            man_n=len(all_manual)
            if man_n>0:
                man_wins=[t for t in all_manual if t["pnl"]>0]
                man_wr=round(len(man_wins)/man_n*100,1)
                man_aw=round(np.mean([t["pnl"] for t in man_wins]),2) if man_wins else 0
                man_total=round(sum(t["pnl"] for t in all_manual),2)
                m1,m2,m3=st.columns(3)
                m1.markdown(kcard("승률",f"{man_wr}%","up" if man_wr>=50 else "down"),unsafe_allow_html=True)
                m2.markdown(kcard("평균 수익",f"+{man_aw}%","up"),unsafe_allow_html=True)
                m3.markdown(kcard("총 손익",f"{man_total:+.1f}%","up" if man_total>=0 else "down"),unsafe_allow_html=True)

                # 비교 바 차트
                if ais["n"]>0:
                    fig_cmp=go.Figure(data=[
                        go.Bar(name="🤖 AI",x=["승률(%)","평균수익(%)"],
                               y=[ais["wr"],ais["aw"]],marker_color="#6366f1"),
                        go.Bar(name="👤 나",x=["승률(%)","평균수익(%)"],
                               y=[man_wr,man_aw],marker_color="#f59e0b"),
                    ])
                    fig_cmp.update_layout(barmode="group",height=220,
                        template="plotly_white",paper_bgcolor="#fff",
                        plot_bgcolor="#fafafa",
                        font=dict(color="#374151",size=11),
                        margin=dict(l=10,r=10,t=10,b=10),
                        legend=dict(orientation="h",y=1.1))
                    st.plotly_chart(fig_cmp,use_container_width=True)
            else:
                st.info("완료된 거래(매수+매도 쌍)를 '거래 입력' 탭에서 추가하세요.")
        st.markdown("---")

# ── 탭5: 백테스트 ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="stitle">🔬 백테스트 결과</div>',
                unsafe_allow_html=True)
    if not bt_t and not bt_b:
        st.info("사이드바에서 **백테스트 실행** 버튼을 눌러주세요.")
    else:
        # 비교 차트
        fig_cmp2=go.Figure()
        if bt_t:
            eq=bt_t["eq"]
            fig_cmp2.add_trace(go.Scatter(x=eq.index,y=(eq/eq.iloc[0]*100).round(2),
                name=f"TQQQ {tqqq_ver}",line=dict(color="#2563eb",width=2)))
        if bt_b:
            eq=bt_b["eq"]
            fig_cmp2.add_trace(go.Scatter(x=eq.index,y=(eq/eq.iloc[0]*100).round(2),
                name=f"BULZ {bulz_ver}",line=dict(color="#ea580c",width=2.2)))
        fig_cmp2.update_layout(**WL,title="두 전략 수익률 비교 (기준=100)",
                               height=340,hovermode="x unified",
                               legend=dict(orientation="h",y=1.08))
        fig_cmp2.update_xaxes(gridcolor="#e2e8f0")
        fig_cmp2.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig_cmp2,use_container_width=True)

        for tk,res,color,ver in [("TQQQ",bt_t,"#2563eb",tqqq_ver),
                                  ("BULZ",bt_b,"#ea580c",bulz_ver)]:
            if not res: continue
            s=res["s"]
            st.markdown(f'<span style="color:{color};font-size:15px;font-weight:700;">'
                        f'{tk}</span> — <span class="ver-badge">{ver}</span>',
                        unsafe_allow_html=True)
            r1,r2,r3,r4,r5,r6=st.columns(6)
            r1.markdown(kcard("총 수익률",f"{s['ret']:+.1f}%","up" if s["ret"]>=0 else "down"),unsafe_allow_html=True)
            r2.markdown(kcard("CAGR",f"{s['cagr']:+.1f}%","up" if s["cagr"]>=0 else "down"),unsafe_allow_html=True)
            r3.markdown(kcard("MDD",f"{s['mdd']:.1f}%","warn"),unsafe_allow_html=True)
            r4.markdown(kcard("샤프",str(s["sharpe"]),"up" if s["sharpe"]>=1 else "neu"),unsafe_allow_html=True)
            r5.markdown(kcard("승률",f"{s['wr']:.1f}%","up" if s["wr"]>=50 else "down"),unsafe_allow_html=True)
            r6.markdown(kcard("거래수",f"{s['trades']}회"),unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            st.plotly_chart(chart_bt(res),use_container_width=True)
            with st.expander(f"{tk} 매매 내역 전체"):
                if res["trades"]:
                    st.dataframe(pd.DataFrame(res["trades"]),
                        use_container_width=True,hide_index=True,
                        height=min(38*len(res["trades"])+38,300))
            st.markdown("---")

# ── 탭6: 모델 버전 관리 ───────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="stitle">📈 모델 버전 관리</div>',
                unsafe_allow_html=True)
    versions=st.session_state.model_versions
    if not versions:
        st.info("아직 업데이트된 버전이 없어요.\n\n"
                "사이드바에서 **'TQQQ/BULZ 모델 업데이트'** 버튼을 누르면 "
                "최신 데이터로 재최적화 후 새 버전이 저장돼요.")
    else:
        st.plotly_chart(chart_version_perf(),use_container_width=True)
        st.markdown("#### 버전 상세 기록")
        for ver in reversed(versions):
            tk=ver["ticker"]; color=TC[tk]
            s=ver["stats"]
            with st.expander(f'{tk} {ver["version"]}  —  '
                             f'수익률 {s["ret"]:+.1f}%  승률 {s["wr"]:.1f}%  '
                             f'샤프 {s["sharpe"]:.2f}  ({ver["date"]})'):
                st.markdown(f"**데이터**: {ver['data_days']}일  |  **노트**: {ver['note']}")
                st.markdown("**사용 파라미터**")
                p=ver["params"]
                pc1,pc2,pc3=st.columns(3)
                pc1.markdown(kcard("EMA",f"{p['ema_s']}/{p['ema_l']}"),unsafe_allow_html=True)
                pc2.markdown(kcard("RSI",f"{p['rsi_p']}일 ({p['rl']}~{p['rh']})"),unsafe_allow_html=True)
                pc3.markdown(kcard("MACD",f"{p['mf']}/{p['ms']}"),unsafe_allow_html=True)
                ps1,ps2,ps3=st.columns(3)
                ps1.markdown(kcard("손절",f"-{int(p['sl']*100)}%"),unsafe_allow_html=True)
                ps2.markdown(kcard("트레일링",f"-{int(p.get('tr',0)*100)}%" if p.get('tr') else "없음"),unsafe_allow_html=True)
                ps3.markdown(kcard("ATR배수",str(p.get('atr_m','없음'))),unsafe_allow_html=True)
                st.markdown("**성과**")
                rs1,rs2,rs3,rs4=st.columns(4)
                rs1.markdown(kcard("수익률",f"{s['ret']:+.1f}%","up" if s["ret"]>=0 else "down"),unsafe_allow_html=True)
                rs2.markdown(kcard("MDD",f"{s['mdd']:.1f}%","warn"),unsafe_allow_html=True)
                rs3.markdown(kcard("승률",f"{s['wr']:.1f}%","up" if s["wr"]>=50 else "down"),unsafe_allow_html=True)
                rs4.markdown(kcard("종합점수",f"{ver['score']}"),unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 이 앱은 교육 목적입니다. 투자 권유가 아닙니다.")
