# app.py — Versión optimizada para Streamlit Cloud
# - Gemini 2.0 Pro (modelo cacheado)
# - Análisis unificado de noticias + resumen + contexto (1 llamada)
# - Métricas extra: CAGR, Sortino Ratio
# - Mejoras de cache, UX y estabilidad
# Requerimientos: streamlit, yfinance, pandas, numpy, plotly, statsmodels, feedparser, google-generative-ai

import time
from datetime import datetime, timedelta
import urllib.parse
import logging

import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import requests
import statsmodels.api as sm

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generative AI
import google.generativeai as genai

# Optional niceties
from typing import Optional, Dict, List, Tuple, Any

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Analizador Optimizado de Acciones & ETFs", layout="wide")
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("app")

# Load Gemini API key from secrets (recommended) or fall back to placeholder
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
if not GEMINI_KEY:
    st.warning("No se encontró GEMINI_API_KEY en st.secrets. Agregala para habilitar análisis con IA.")
# Don't configure if key missing (avoids exception)
try:
    genai.configure(api_key=GEMINI_KEY or "INVALID_OR_MISSING_KEY")
except Exception as e:
    LOG.warning("No se pudo configurar genai: %s", e)

# -------------------------
# CSS (UI tuning)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #D0D4D7;
    background-color: #0E1117;
}

:root {
    --primary: #4A90E2;
    --secondary: #1A1D23;
}

.card {
    background-color: #1A1D23;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
}

.card:hover {
    border-color: var(--primary);
    background-color: #23262E;
    transform: scale(1.01);
    box-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
}

.small {
    font-size: 12px;
    color: #99A0A6;
}

/* Buttons */
.stButton>button {
    background-color: var(--primary);
    color: white;
    padding: 0.4em 1em;
    border-radius: 6px;
    border: none;
    font-weight: 600;
    transition: background 0.3s ease, transform 0.2s ease;
}

.stButton>button:hover {
    background-color: #357ABD;
    transform: scale(1.03);
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background-color: #14171C;
    border-right: 1px solid rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)


# -------------------------
# Helpers / Formatting
# -------------------------
def fmt_pct(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{x:.{nd}%}"
    except Exception:
        return str(x)

def fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def fmt_bil(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        if abs(x) >= 1e12:
            return f"{x/1e12:.2f} T"
        if abs(x) >= 1e9:
            return f"{x/1e9:.2f} B"
        if abs(x) >= 1e6:
            return f"{x/1e6:.2f} M"
        return f"{x:.0f}"
    except Exception:
        return "—"

# -------------------------
# Financial metrics
# -------------------------
def calculate_period_performance(hist_df: pd.DataFrame) -> Optional[float]:
    """(last / first) - 1"""
    if hist_df is None or hist_df.empty or "Close" not in hist_df.columns or len(hist_df) < 2:
        return None
    try:
        first = float(hist_df["Close"].iloc[0])
        last = float(hist_df["Close"].iloc[-1])
        if first > 0:
            return (last / first) - 1.0
    except Exception:
        return None
    return None

def calculate_cagr(hist_df: pd.DataFrame) -> Optional[float]:
    """Compound Annual Growth Rate over the history period"""
    if hist_df is None or hist_df.empty or "Close" not in hist_df.columns or len(hist_df) < 2:
        return None
    try:
        first = float(hist_df["Close"].iloc[0])
        last = float(hist_df["Close"].iloc[-1])
        days = (hist_df.index[-1] - hist_df.index[0]).days
        if first > 0 and days > 0:
            years = days / 365.25
            return (last / first) ** (1.0 / years) - 1.0
    except Exception:
        return None
    return None

def risk_metrics(close: pd.Series, rf_annual: float = 0.0) -> Dict[str, Optional[float]]:
    """Return dictionary of risk metrics including Sharpe, Sortino, Volatility, Max Drawdown, Calmar, CAGR"""
    out = {"Sharpe": None, "Sortino": None, "Volatilidad Anual": None, "Máx. Drawdown": None, "Calmar": None, "CAGR": None}
    if close is None or close.size < 2:
        return out
    ret = close.pct_change().dropna()
    if ret.empty:
        return out
    ann = 252
    mu = ret.mean() * ann
    sigma = ret.std(ddof=0) * np.sqrt(ann)
    out["Retorno Anualizado"] = mu
    out["Volatilidad Anual"] = sigma
    out["Sharpe"] = (mu - rf_annual) / sigma if sigma and np.isfinite(sigma) else np.nan

    # Sortino (downside deviation)
    downside = ret[ret < 0]
    if not downside.empty:
        downside_std = downside.std(ddof=0) * np.sqrt(ann)
        out["Sortino"] = (mu - rf_annual) / downside_std if downside_std and np.isfinite(downside_std) else np.nan
    else:
        out["Sortino"] = np.nan

    # Drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    max_dd = dd.min()
    out["Máx. Drawdown"] = max_dd
    out["Calmar"] = mu / abs(max_dd) if max_dd and max_dd < 0 else np.nan

    out["CAGR"] = None
    try:
        out["CAGR"] = calculate_cagr(close.to_frame())
    except Exception:
        out["CAGR"] = None

    return out

def compute_beta(asset_close: pd.Series, bench_close: pd.Series) -> Optional[float]:
    if asset_close is None or bench_close is None:
        return None
    ret_a = asset_close.pct_change().dropna()
    ret_b = bench_close.pct_change().dropna()
    df = pd.concat([ret_a, ret_b], axis=1).dropna()
    if df.empty:
        return None
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var_b = np.var(df.iloc[:, 1])
    return cov / var_b if var_b else None

def capm_alpha(asset_close: pd.Series, bench_close: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    r_a = asset_close.pct_change().dropna()
    r_b = bench_close.pct_change().dropna()
    df = pd.concat([r_a, r_b], axis=1).dropna()
    if df.empty:
        return None, None, None
    X = sm.add_constant(df.iloc[:,1])
    y = df.iloc[:,0]
    model = sm.OLS(y, X).fit()
    alpha = model.params.get("const")
    beta = model.params.iloc[1] if len(model.params) > 1 else None
    r2 = model.rsquared
    return alpha, beta, r2

# -------------------------
# Robust download helpers
# -------------------------
def _retry(fn, *args, retries: int = 3, delay: float = 0.8, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(delay * (i + 1))
    raise last

def download_history_single(ticker: str, period: str, interval: str) -> pd.DataFrame:
    hist = _retry(
        yf.download, tickers=ticker, period=period, interval=interval,
        progress=False, auto_adjust=True, threads=False
    )
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0] for c in hist.columns]
        hist = hist.dropna(subset=["Close"])
        hist.sort_index(inplace=True)
        return hist
    return pd.DataFrame()

def download_history_batch(tickers: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = _retry(
        yf.download, tickers=tickers, period=period, interval=interval,
        progress=False, auto_adjust=True, group_by='ticker', threads=True
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(raw, pd.DataFrame) and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        for tk in tickers:
            if tk in first_level:
                df = raw[tk].copy()
                df.columns = [c[0] for c in df.columns] if isinstance(df.columns, pd.MultiIndex) else df.columns
                df = df.dropna(subset=["Close"])
                df.sort_index(inplace=True)
                out[tk] = df
            else:
                out[tk] = pd.DataFrame()
    else:
        for tk in tickers:
            try:
                out[tk] = download_history_single(tk, period, interval)
            except Exception:
                out[tk] = pd.DataFrame()
    return out

# -------------------------
# Streamlit caches
# -------------------------
@st.cache_resource
def get_gemini_model():
    """Cache the model object (initializes once)."""
    try:
        # Use "gemini-2.0-pro" as requested
        model = genai.GenerativeModel("gemini-2.0-pro")
        return model
    except Exception as e:
        LOG.warning("No se pudo inicializar Gemini: %s", e)
        return None

@st.cache_resource
def get_ticker_obj(ticker: str):
    return yf.Ticker(ticker)

@st.cache_data(ttl=7200, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    t = get_ticker_obj(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}
    try:
        fi = t.fast_info or {}
        for k, v in fi.items():
            if k not in info or info.get(k) is None:
                info[k] = v
    except Exception:
        pass
    # try minimal derived ratios
    try:
        fin = t.financials
        bs = t.balance_sheet
        if not fin.empty and not bs.empty:
            income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else None
            equity = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
            assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else None
            if income is not None and equity:
                info["returnOnEquity"] = float(income / equity)
            if income is not None and assets:
                info["returnOnAssets"] = float(income / assets)
    except Exception:
        pass
    return info

@st.cache_data(ttl=7200, show_spinner=False)
def get_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        return download_history_single(ticker, period, interval)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=7200, show_spinner=False)
def get_histories_batch(tickers: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    try:
        return download_history_batch(tickers, period, interval)
    except Exception:
        return {tk: pd.DataFrame() for tk in tickers}

# -------------------------
# News fetching & single-model analysis
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_news(query: str, days: int = 7, max_items_per_topic: int = 6) -> List[Dict[str, Any]]:
    """Fetch news from Google News RSS for micro + macro topics and dedupe."""
    if not query:
        return []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    encoded_query = urllib.parse.quote_plus(f"{query} stock OR finance OR earnings")

    def fetch_feed(search_query):
        feed_url = (
            f"https://news.google.com/rss/search?q={search_query}+after:{start_date.strftime('%Y-%m-%d')}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(feed_url)
            out = []
            for e in feed.entries[:max_items_per_topic]:
                published = getattr(e, "published", "")
                summary = getattr(e, "summary", "")
                link = getattr(e, "link", "")
                title = getattr(e, "title", "")
                out.append({"title": title, "link": link, "published": published, "summary": summary})
            return out
        except Exception:
            return []

    micro = fetch_feed(encoded_query)
    macro_topics = [
        "US Federal Reserve interest rates",
        "inflation CPI United States",
        "global economic outlook IMF",
        "oil prices energy market",
        "US China trade tariffs",
        "stock market volatility investor sentiment",
    ]
    macro = []
    for t in macro_topics:
        macro.extend(fetch_feed(urllib.parse.quote_plus(t)))

    combined = micro + macro
    seen = set()
    dedup = []
    for n in combined:
        title = n.get("title") or ""
        if title not in seen:
            dedup.append(n)
            seen.add(title)
    return dedup[: max_items_per_topic * 3]

@st.cache_data(ttl=1800, show_spinner=False)
def analyze_news_and_generate_advice(ticker: str, news_list: List[Dict[str, Any]], perf_str: str, risk_summary: str) -> str:
    """
    Single call to Gemini:
    - provide news list
    - return: market sentiment, explanation, macro summary, and 3-line executive summary
    """
    model = get_gemini_model()
    if model is None:
        return "⚠️ Gemini no está disponible. Asegurate de configurar tu API key en st.secrets['GEMINI_API_KEY']."

    # Compose compact news content (trim summaries to avoid huge prompts)
    def safe_truncate(s, n=360):
        return (s[:n] + "...") if s and len(s) > n else (s or "")

    content_lines = []
    for n in news_list[:20]:
        title = n.get("title", "")
        summary = safe_truncate(n.get("summary", ""), 300)
        published = n.get("published", "")
        content_lines.append(f"- {title} ({published}): {summary}")
    content_text = "\n".join(content_lines) if content_lines else "No news found."

    prompt = f"""
You are an expert financial analyst. You will be given recent news items and some numeric context for a stock ticker.

Ticker: {ticker}
Price/Perf (user context): {perf_str}
Risk metrics summary: {risk_summary}

NEWS (most recent first):
{content_text}

Task (produce a single response in Spanish, concise):
1) Provide a short market sentiment for the ticker: Positivo / Negativo / Neutro and a 1–2 sentence rationale referencing the most important news items.
2) Give a 3-sentence macroeconomic summary relevant to equity markets (rates, inflation, commodities, investor sentiment).
3) Provide a 3-line executive recommendation oriented to a quant/PM audience:
   - One-line situational summary
   - One-line key risks/opportunities
   - One-line verdict: Positivo / Neutro / Negativo

Be factual, cite the 1–2 news titles that most support your claims inline (just the title text). Keep the whole reply under ~400 words.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        LOG.exception("Error al llamar a Gemini: %s", e)
        return f"⚠️ Error generando análisis con Gemini: {e}"

# -------------------------
# Core analysis function (refactor of analizar_ticker)
# -------------------------
def is_etf(info: dict) -> bool:
    try:
        qt = (info.get("quoteType") or "").lower()
        name = (info.get("shortName") or info.get("longName") or "").lower()
        return "etf" in qt or "etf" in name
    except Exception:
        return False

def analyze_ticker(
    ticker: str,
    pe_thr: float, roe_thr: float, pb_thr: float,
    w_pe: float, w_roe: float, w_pb: float
) -> Tuple[pd.DataFrame, str, List[str], Dict[str, Any], Any]:
    t = get_ticker_obj(ticker)
    info = get_ticker_info(ticker)

    def safe_pick(info_dict: dict, *keys, default=None, scale=None):
        for k in keys:
            v = info_dict.get(k)
            if v is not None and pd.notna(v):
                return v * scale if (scale is not None and v is not None) else v
        return default

    market_cap = safe_pick(info, "marketCap", "market_cap")
    trailing_pe = safe_pick(info, "trailingPE", "trailing_pe")
    forward_pe = safe_pick(info, "forwardPE", "forward_pe")
    pb = safe_pick(info, "priceToBook", "price_to_book")
    ev_ebitda = safe_pick(info, "enterpriseToEbitda", "ev_to_ebitda")
    roe = safe_pick(info, "returnOnEquity", "return_on_equity", scale=1.0)
    roa = safe_pick(info, "returnOnAssets", "return_on_assets", scale=1.0)
    profit_margins = safe_pick(info, "profitMargins", "profit_margins", scale=1.0)
    dividend_yield = safe_pick(info, "dividendYield", "dividend_yield", scale=1.0)
    expense_ratio = safe_pick(info, "annualReportExpenseRatio", scale=1.0)

    currency = safe_pick(info, "currency") or safe_pick(info, "financialCurrency") or "—"
    exchange = safe_pick(info, "exchange") or "—"
    sector = safe_pick(info, "sector") or "—"
    industry = safe_pick(info, "industry") or "—"
    long_name = safe_pick(info, "longName") or safe_pick(info, "shortName") or ticker

    data = {
        "Market Cap": [fmt_bil(market_cap)],
        "P/E Trailing": [fmt_num(trailing_pe)],
        "P/E Forward": [fmt_num(forward_pe)],
        "P/B": [fmt_num(pb)],
        "EV/EBITDA": [fmt_num(ev_ebitda)],
        "ROE": [fmt_pct(roe)],
        "ROA": [fmt_pct(roa)],
        "Profit Margins": [fmt_pct(profit_margins)],
        "Dividend Yield": [fmt_pct(dividend_yield)],
        "Expense Ratio": [fmt_pct(expense_ratio)],
        "Moneda": [currency],
        "Exchange": [exchange],
        "Sector": [sector],
        "Industria": [industry],
    }
    df = pd.DataFrame(data)

    razones, scores = [], []
    if trailing_pe is not None and pd.notna(trailing_pe):
        cond = trailing_pe < pe_thr
        razones.append(f"P/E Trailing = {trailing_pe:.2f} → {'Bajo (atractivo)' if cond else 'Alto (menos atractivo)'} (umbral {pe_thr})")
        scores.append(w_pe if cond else 0.0)
    else:
        razones.append("P/E Trailing no disponible")

    if roe is not None and pd.notna(roe):
        cond = roe > roe_thr
        razones.append(f"ROE = {roe:.2%} → {'Alto (positivo)' if cond else 'Bajo (menos atractivo)'} (umbral {roe_thr:.0%})")
        scores.append(w_roe if cond else 0.0)
    else:
        razones.append("ROE no disponible")

    if pb is not None and pd.notna(pb):
        try:
            pb_float = float(pb)
            cond = pb_float < pb_thr
            razones.append(f"P/B = {pb_float:.2f} → {'Atractivo' if cond else 'Elevado'} (umbral {pb_thr})")
            scores.append(w_pb if cond else 0.0)
        except Exception:
            razones.append("P/B no numérico")
    else:
        razones.append("P/B no disponible")

    total_w = w_pe + w_roe + w_pb
    score = (sum(scores) / total_w) if total_w > 0 else 0.0
    decision = "✅ Barato" if score >= 0.67 else ("🟡 Neutro" if score >= 0.33 else "❌ Caro")

    nota = None
    ietf = is_etf(info)
    if ietf:
        nota = "Este ticker parece ser un ETF: algunos ratios (P/E, ROE, P/B) pueden no estar disponibles o no ser comparables."

    meta = {
        "long_name": long_name,
        "currency": currency,
        "exchange": exchange,
        "sector": sector,
        "industry": industry,
        "is_etf": ietf,
        "nota": nota,
        "score": score,
    }

    return df, decision, razones, meta, t

# -------------------------
# UI: Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Parámetros")
    col1, col2 = st.columns([3,1])
    with col1:
        ticker = st.text_input("Ticker (ej: AAPL, MSFT, SPY)", "AAPL").strip().upper()
    with col2:
        refresh = st.button("🔄 Reset cache & rerun")

    if refresh:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Cache limpiada. Recargando...", icon="♻️")
        st.experimental_rerun()

    st.subheader("Umbrales")
    pe_thr = st.number_input("Umbral P/E (trailing) máximo", min_value=1.0, max_value=200.0, value=15.0, step=0.5)
    roe_thr = st.number_input("Umbral ROE mínimo", min_value=0.0, max_value=2.0, value=0.15, step=0.01)
    pb_thr = st.number_input("Umbral P/B máximo", min_value=0.1, max_value=50.0, value=2.0, step=0.1)

    st.subheader("Pesos (total relevante)")
    w_pe = st.slider("Peso P/E", 0.0, 1.0, 0.34)
    w_roe = st.slider("Peso ROE", 0.0, 1.0, 0.33)
    w_pb = st.slider("Peso P/B", 0.0, 1.0, 0.33)

    st.subheader("Serie de precios")
    periodo = st.selectbox("Período", ["1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], index=2)
    intervalo = st.selectbox("Intervalo", ["1d","5d","1wk","1mo","3mo"], index=0)
    sma50_on = st.checkbox("Mostrar SMA 50", value=True)
    sma200_on = st.checkbox("Mostrar SMA 200", value=False)
    rsi_on = st.checkbox("Mostrar RSI", value=False)
    log_scale = st.checkbox("Escala logarítmica", value=False)

    st.subheader("Benchmark")
    benchmark = st.selectbox("Benchmark para Beta", ["SPY", "QQQ", "EFA", "IWM"], index=0)

    st.subheader("Análisis con IA")
    ai_enabled = st.checkbox("🧠 Activar análisis con Gemini", value=True)

# -------------------------
# Main layout: Tabs
# -------------------------
st.title("📈 Analizador Optimizado de Acciones & ETFs")

tab1, tab2 = st.tabs(["🔍 Análisis individual", "🏁 Comparar Tickers"])

with tab1:
    if not ticker:
        st.info("Ingresá un ticker en la barra lateral para comenzar.")
    else:
        # Obtain fundamental analysis and history
        df_fund, decision, razones, meta, tk_obj = analyze_ticker(
            ticker, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb
        )
        hist = get_history(ticker, periodo, intervalo)

        st.subheader(f"🪪 {meta['long_name']} ({ticker})")

        # Top-level price metrics
        period_perf = None
        last_close = None
        max_p = None
        min_p = None
        last_vol = None
        avg_vol = None
        vol_delta = None
        plot_df = pd.DataFrame()

        if not hist.empty:
            plot_df = hist.copy()
            period_perf = calculate_period_performance(plot_df)
            last_close = plot_df["Close"].iloc[-1]
            max_p = plot_df["High"].max() if "High" in plot_df.columns else None
            min_p = plot_df["Low"].min() if "Low" in plot_df.columns else None
            last_vol = plot_df["Volume"].iloc[-1] if "Volume" in plot_df.columns else None

            # vol SMA50
            if 'Volume' in plot_df.columns and len(plot_df) >= 50:
                plot_df["Vol_SMA50"] = plot_df["Volume"].rolling(50).mean()
                avg_vol = plot_df["Vol_SMA50"].iloc[-1] if pd.notna(plot_df["Vol_SMA50"].iloc[-1]) else None
                if avg_vol and last_vol:
                    vol_delta = (last_vol / avg_vol) - 1.0

            # metrics panel
            st.write("---")
            def card(title, value, subtitle=None, color="#4A90E2", icon="📈"):
                st.markdown(
                    f"""
                    <div class="card">
                        <div style="font-size:18px">{icon}</div>
                        <div style="font-weight:600; font-size:14px; color:#cfd8dc">{title}</div>
                        <div style="font-size:20px; font-weight:700; color:{color}">{value}</div>
                        <div class="small">{subtitle or ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                card("Precio Actual", f"{last_close:.2f}" if last_close is not None else "—", "Cierre más reciente", icon="💰")
            with c2:
                card("Cambio", fmt_pct(period_perf) if period_perf is not None else "—", f"en {periodo}", icon="📊",
                     color="limegreen" if (period_perf and period_perf > 0) else "tomato")
            with c3:
                card("Máximo", f"{max_p:.2f}" if max_p is not None else "—", "del período", icon="📈")
            with c4:
                card("Mínimo", f"{min_p:.2f}" if min_p is not None else "—", "del período", icon="📉")
            with c5:
                card("Volumen", fmt_bil(last_vol), "último día", icon="📦")

            st.write("---")

            # fundamental meta cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("Resultado Fundamental", decision, icon="🧮", color="#FFD700")
            with c2:
                card("Puntaje", f"{meta['score']*100:.0f} %", "Evaluación combinada", icon="⭐")
            with c3:
                card("Market Cap", fmt_bil(get_ticker_info(ticker).get("marketCap")), "Capitalización", icon="🏦")
            with c4:
                card("Moneda / Exchange", f"{meta['currency']} · {meta['exchange']}", "", icon="💱")

            st.write("---")

            # risk metrics
            if not plot_df.empty:
                rm = risk_metrics(plot_df["Close"])
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: card("Sharpe", fmt_num(rm.get("Sharpe")), "Ratio rendimiento/riesgo", icon="⚖️")
                with c2: card("Sortino", fmt_num(rm.get("Sortino")), "Solo penaliza caídas", icon="🔻")
                with c3: card("Máx. Drawdown", fmt_pct(rm.get("Máx. Drawdown")), "Caída máxima", icon="📉", color="orange")
                with c4: card("CAGR", fmt_pct(rm.get("CAGR")), "Rendimiento anual compuesto", icon="📈", color="limegreen")
                with c5: card("Vol. Anual", fmt_pct(rm.get("Volatilidad Anual")), "Desviación estándar anual", icon="🌪️")

            st.write("---")

            # CAPM / Beta
            if not plot_df.empty and benchmark:
                bench_hist = get_history(benchmark, periodo, intervalo)
                if not bench_hist.empty:
                    beta_val = compute_beta(plot_df["Close"], bench_hist["Close"])
                    try:
                        alpha, beta_ols, r2 = capm_alpha(plot_df["Close"], bench_hist["Close"])
                    except Exception:
                        alpha, beta_ols, r2 = None, None, None
                    c1, c2, c3 = st.columns(3)
                    with c1: card("β vs Benchmark", fmt_num(beta_val), f"vs {benchmark}", icon="📉")
                    with c2: card("α (CAPM)", fmt_pct(alpha), "Rendimiento extra", icon="⚡", color="#FFD700")
                    with c3: card("R² (CAPM)", fmt_num(r2), "Ajuste CAPM", icon="📐", color="#B0E0E6")

        # ETF note
        if meta.get("is_etf") and meta.get("nota"):
            st.info(meta["nota"])

        # period performance summary
        if period_perf is not None:
            if period_perf > 0:
                st.success(f"📈 El precio ha subido un **{period_perf:.2%}** en el período {periodo}")
            elif period_perf < 0:
                st.error(f"📉 El precio ha bajado un **{abs(period_perf):.2%}** en el período {periodo}")
            else:
                st.info(f"➡️ El precio se mantiene sin cambios en el período {periodo}")

        # Fundamental table
        st.subheader("📊 Ratios Fundamentales")
        order_cols = [
            "Market Cap", "Dividend Yield", "Expense Ratio",
            "P/E Trailing", "P/E Forward", "P/B", "EV/EBITDA",
            "ROE", "ROA", "Profit Margins",
            "Moneda", "Exchange", "Sector", "Industria"
        ]
        df_fund = df_fund[[c for c in order_cols if c in df_fund.columns]]
        st.dataframe(df_fund, use_container_width=True)

        # Evaluacion fundamental reasons
        st.subheader("📌 Evaluación Fundamental")
        for r in razones:
            st.write(f"- {r}")

        # Technical analysis charts
        st.subheader("📉 Análisis Técnico y Evolución del Precio")
        if plot_df.empty:
            st.error("No hay datos históricos disponibles para los parámetros seleccionados.")
        else:
            # SMA
            if sma50_on and len(plot_df) >= 50:
                plot_df["SMA50"] = plot_df["Close"].rolling(50).mean()
            if sma200_on and len(plot_df) >= 200:
                plot_df["SMA200"] = plot_df["Close"].rolling(200).mean()

            # simple trendline via peaks (best-effort)
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(-plot_df["Close"], distance=10)
                if len(peaks) >= 2:
                    coeffs = np.polyfit(peaks, plot_df["Close"].iloc[peaks].values, 1)
                    plot_df["Trendline"] = coeffs[0] * np.arange(len(plot_df)) + coeffs[1]
            except Exception:
                pass

            # crosses
            golden_crosses = []
            death_crosses = []
            if "SMA50" in plot_df.columns and "SMA200" in plot_df.columns:
                prev50 = plot_df["SMA50"].shift(1)
                prev200 = plot_df["SMA200"].shift(1)
                gc_mask = (plot_df["SMA50"] > plot_df["SMA200"]) & (prev50 <= prev200)
                dc_mask = (plot_df["SMA50"] < plot_df["SMA200"]) & (prev50 >= prev200)
                golden_crosses = plot_df.index[gc_mask].tolist()
                death_crosses = plot_df.index[dc_mask].tolist()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                         low=plot_df['Low'], close=plot_df['Close'], name='Precio'), row=1, col=1)
            for col in ["SMA50", "SMA200", "Trendline"]:
                if col in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], name=col, line=dict(width=2)), row=1, col=1)
            if golden_crosses and "SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=golden_crosses, y=plot_df.loc[golden_crosses]["SMA50"],
                                         name='Golden Cross', mode='markers', marker=dict(color='gold', size=12, symbol='star')), row=1, col=1)
            if death_crosses and "SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=death_crosses, y=plot_df.loc[death_crosses]["SMA50"],
                                         name='Death Cross', mode='markers', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)

            if 'Volume' in plot_df.columns:
                fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Volumen', marker_color='lightgrey'), row=2, col=1)
            if "Vol_SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol_SMA50"], name='Vol SMA(50)', line=dict(width=1, color='slategray')), row=2, col=1)

            fig.update_layout(template="plotly_dark", title=f"Análisis Técnico - {ticker} | Cambio {periodo}: {fmt_pct(period_perf)}", xaxis_rangeslider_visible=False, height=640)
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showspikes=True, spikemode="across")
            fig.update_yaxes(title_text=f"Precio ({meta['currency']})", row=1, col=1, type="log" if log_scale else "linear")
            fig.update_yaxes(title_text="Volumen", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # signals
            last_signal = None
            if golden_crosses:
                last_signal = ("golden", golden_crosses[-1])
            if death_crosses and (not golden_crosses or (death_crosses and death_crosses[-1] > golden_crosses[-1])):
                last_signal = ("death", death_crosses[-1])
            if last_signal:
                typ, dt = last_signal
                if typ == "golden":
                    st.success(f"✨ Golden Cross detectado el {pd.to_datetime(dt).strftime('%Y-%m-%d')}")
                else:
                    st.error(f"💀 Death Cross detectado el {pd.to_datetime(dt).strftime('%Y-%m-%d')}")
            elif sma50_on and sma200_on:
                st.info("No se detectaron cruces 50/200 en el período seleccionado.")

            # RSI
            if rsi_on and len(plot_df) > 14:
                close = plot_df["Close"]
                delta = close.diff()
                roll_up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                roll_down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                rs = roll_up / roll_down
                rsi = 100 - (100 / (1 + rs))
                rsi_df = pd.DataFrame({"RSI": rsi}).dropna()
                if not rsi_df.empty:
                    fig_rsi = px.line(rsi_df, x=rsi_df.index, y="RSI", title=f"RSI (14) - {ticker}", template="plotly_dark")
                    fig_rsi.update_layout(xaxis_title="Fecha", yaxis_title="RSI")
                    fig_rsi.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.12)
                    fig_rsi.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.12)
                    st.plotly_chart(fig_rsi, use_container_width=True)

        # AI analysis (single combined call)
        if ai_enabled:
            st.write("---")
            st.subheader("🧠 Análisis AI unificado (noticias + resumen + macro)")

            with st.spinner(f"Buscando noticias y generando análisis con Gemini..."):
                query_name = meta.get("long_name") or ticker
                news = fetch_recent_news(query_name, days=7, max_items_per_topic=6)
                # Prepare a compact risk summary
                risk_desc = ""
                try:
                    if not plot_df.empty:
                        rm = risk_metrics(plot_df["Close"])
                        risk_desc = f"Sharpe {fmt_num(rm.get('Sharpe'))}, Sortino {fmt_num(rm.get('Sortino'))}, MáxDD {fmt_pct(rm.get('Máx. Drawdown'))}, CAGR {fmt_pct(rm.get('CAGR'))}"
                    else:
                        risk_desc = "No historical risk metrics available."
                except Exception:
                    risk_desc = "No historical risk metrics available."

                perf_str = f"{fmt_pct(period_perf)} over {periodo}" if period_perf is not None else "—"

                ai_text = analyze_news_and_generate_advice(ticker, news, perf_str, risk_desc)

            if news:
                with st.expander("🗞️ Ver noticias recientes"):
                    for n in news:
                        st.markdown(f"- [{n['title']}]({n['link']})  \n<sub>{n['published']}</sub>", unsafe_allow_html=True)
            else:
                st.info("No se encontraron noticias recientes.")

            st.markdown(ai_text)

with tab2:
    st.header("📊 Comparador de Tickers")
    st.write("Ingresa una lista de tickers separados por comas. Ej: `AAPL, MSFT, GOOG, JPM, BAC`")
    tickers_input = st.text_area("Tickers", value="AAPL, MSFT, GOOG, JPM, BAC").strip()

    analysis_mode = st.radio(
        "Modo de Análisis",
        ("Contexto Sectorial (Recomendado)", "Umbrales Absolutos (Original)"),
        horizontal=True
    )
    st.caption("El comparador usa los pesos, período e intervalo definidos en la barra lateral.")

    if not tickers_input:
        st.info("Ingresa al menos un ticker para comenzar el análisis.")
    else:
        tickers = []
        for t in tickers_input.split(","):
            t = t.strip().upper()
            if t and t not in tickers:
                tickers.append(t)

        if len(tickers) == 0:
            st.warning("La lista de tickers está vacía.")
        else:
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                sort_metric = st.selectbox("Ordenar por", ["Puntaje", f"Perf ({periodo})", "Volatilidad (Anual)", "Market Cap", "ROE", "P/E Trailing", "P/B"], index=0)
            with c2:
                asc = st.checkbox("Ascendente", value=False)
            with c3:
                mcap_min = st.number_input("Market Cap mínimo (USD, millones)", min_value=0, value=0, step=100)
                mcap_min = mcap_min * 1_000_000

            if analysis_mode == "Umbrales Absolutos (Original)":
                progress = st.progress(0.0, text="Analizando con umbrales absolutos…")
                rows = []
                failed = []
                for i, tk in enumerate(tickers, start=1):
                    try:
                        df_i, decision_i, _, meta_i, _ = analyze_ticker(tk, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb)
                        hist_i = get_history(tk, periodo, intervalo)
                        perf = calculate_period_performance(hist_i)
                        volatilidad = None
                        if not hist_i.empty:
                            retornos = hist_i['Close'].pct_change()
                            volatilidad = retornos.std(ddof=0) * np.sqrt(252)
                        try:
                            raw_mcap = get_ticker_info(tk).get("marketCap")
                        except Exception:
                            raw_mcap = None

                        rows.append({
                            "Ticker": tk,
                            "Nombre": meta_i["long_name"],
                            "Resultado": decision_i,
                            "Puntaje": round(meta_i["score"] * 100, 1),
                            "P/E Trailing": df_i.get("P/E Trailing", ["—"])[0] if "P/E Trailing" in df_i else "—",
                            "ROE": df_i.get("ROE", ["—"])[0] if "ROE" in df_i else "—",
                            "P/B": df_i.get("P/B", ["—"])[0] if "P/B" in df_i else "—",
                            "Market Cap": df_i.get("Market Cap", ["—"])[0] if "Market Cap" in df_i else "—",
                            "_MarketCapRaw": raw_mcap,
                            "Sector": meta_i["sector"],
                            f"Perf ({periodo})": fmt_pct(perf),
                            "Volatilidad (Anual)": fmt_pct(volatilidad),
                        })
                    except Exception as e:
                        st.toast(f"Error analizando {tk}: {e}", icon="⚠️")
                        failed.append(tk)
                    finally:
                        progress.progress(i / len(tickers), text=f"Analizando {tk} ({i}/{len(tickers)})")
                progress.empty()

                if failed:
                    st.warning(f"No se pudieron analizar: {', '.join(failed)}")

                if rows:
                    rank_df = pd.DataFrame(rows)
                    if mcap_min > 0 and "_MarketCapRaw" in rank_df:
                        rank_df = rank_df[(rank_df["_MarketCapRaw"].fillna(0) >= mcap_min)]

                    sort_key = sort_metric
                    if sort_metric in [f"Perf ({periodo})", "Volatilidad (Anual)"]:
                        def to_float_pct(x):
                            if isinstance(x, str) and x.endswith("%"):
                                try: return float(x.replace("%",""))/100.0
                                except: return np.nan
                            return x
                        rank_df["_sortcol"] = rank_df[sort_metric].apply(to_float_pct)
                        sort_key = "_sortcol"
                    elif sort_metric == "Market Cap":
                        sort_key = "_MarketCapRaw" if "_MarketCapRaw" in rank_df.columns else "Market Cap"

                    rank_df_sorted = rank_df.sort_values(by=sort_key, ascending=asc, ignore_index=True).drop(columns=["_sortcol"], errors="ignore")
                    st.subheader("🏆 Ranking (Umbrales Absolutos)")
                    st.dataframe(rank_df_sorted.drop(columns=["_MarketCapRaw"], errors="ignore"), use_container_width=True)

                    st.download_button("⬇️ Descargar ranking (CSV)", data=rank_df_sorted.drop(columns=["_MarketCapRaw"], errors="ignore").to_csv(index=False).encode("utf-8"), file_name="ranking_absoluto.csv", mime="text/csv")
                else:
                    st.error("No se obtuvieron resultados válidos.")
            else:
                # Contexto sectorial (recommended)
                progress = st.progress(0.0, text="Fase 1/2: Descargando históricos en batch…")
                hist_map = get_histories_batch(tickers, periodo, intervalo)
                all_data = []
                failed = []
                for i, tk in enumerate(tickers, start=1):
                    try:
                        info = get_ticker_info(tk)
                        hist = hist_map.get(tk, pd.DataFrame())
                        perf = calculate_period_performance(hist)
                        volatilidad = None
                        if not hist.empty:
                            volatilidad = hist['Close'].pct_change().std(ddof=0) * np.sqrt(252)
                        all_data.append({
                            "Ticker": tk,
                            "Nombre": info.get("longName") or info.get("shortName") or tk,
                            "Sector": info.get("sector") or "Desconocido",
                            "P/E Trailing": info.get("trailingPE"),
                            "ROE": info.get("returnOnEquity"),
                            "P/B": info.get("priceToBook"),
                            "Market Cap": info.get("marketCap"),
                            "_MarketCapRaw": info.get("marketCap"),
                            f"Perf ({periodo})": perf,
                            "Volatilidad (Anual)": volatilidad,
                        })
                    except Exception as e:
                        st.toast(f"No se pudieron obtener datos para {tk}: {e}", icon="⚠️")
                        failed.append(tk)
                    finally:
                        progress.progress(i / len(tickers), text=f"Fase 1/2: Procesando {tk}")
                progress.empty()
                if failed:
                    st.warning(f"No se pudieron analizar: {', '.join(failed)}")

                master_df = pd.DataFrame(all_data).dropna(subset=['Sector']).reset_index(drop=True)
                if mcap_min > 0 and "_MarketCapRaw" in master_df.columns:
                    master_df = master_df[(master_df["_MarketCapRaw"].fillna(0) >= mcap_min)]

                if not master_df.empty:
                    sectores = sorted([s for s in master_df["Sector"].dropna().unique() if s])
                    sel_sectores = st.multiselect("Filtrar sectores", options=sectores, default=sectores)
                    master_df = master_df[master_df["Sector"].isin(sel_sectores)] if sel_sectores else master_df

                    progress = st.progress(1.0, text="Fase 2/2: Analizando contexto sectorial…")
                    metrics_to_average = ["P/E Trailing", "ROE", "P/B"]
                    def robust_sector_stat(x: pd.Series):
                        x_w = x.dropna()
                        if x_w.empty:
                            return np.nan
                        ql, qh = x_w.quantile([0.01, 0.99])
                        return x_w.clip(lower=ql, upper=qh).median()
                    for metric in metrics_to_average:
                        master_df[f"{metric} (Prom. Sector)"] = master_df.groupby('Sector')[metric].transform(robust_sector_stat)

                    def reevaluate_con_contexto(row):
                        scores = []
                        pe, pe_avg = row.get("P/E Trailing"), row.get("P/E Trailing (Prom. Sector)")
                        if pd.notna(pe) and pd.notna(pe_avg) and pe_avg > 0:
                            scores.append(w_pe if pe < pe_avg else 0)
                            row["P/E vs Sector (%)"] = (pe / pe_avg - 1.0) if pe_avg else np.nan
                        else:
                            row["P/E vs Sector (%)"] = np.nan
                        roe, roe_avg = row.get("ROE"), row.get("ROE (Prom. Sector)")
                        if pd.notna(roe) and pd.notna(roe_avg):
                            scores.append(w_roe if roe > roe_avg else 0)
                            row["ROE vs Sector (pp)"] = (roe - roe_avg) if pd.notna(roe_avg) else np.nan
                        else:
                            row["ROE vs Sector (pp)"] = np.nan
                        pb, pb_avg = row.get("P/B"), row.get("P/B (Prom. Sector)")
                        if pd.notna(pb) and pd.notna(pb_avg) and pb_avg > 0:
                            scores.append(w_pb if pb < pb_avg else 0)
                            row["P/B vs Sector (%)"] = (pb / pb_avg - 1.0) if pb_avg else np.nan
                        else:
                            row["P/B vs Sector (%)"] = np.nan
                        total_w = w_pe + w_roe + w_pb
                        score = (sum(scores) / total_w) if total_w > 0 else 0
                        row["Puntaje"] = round(score * 100, 1)
                        row["Resultado"] = "✅ Atractivo vs Pares" if score >= 0.67 else ("🟡 Neutro vs Pares" if score >= 0.33 else "❌ Caro vs Pares")
                        return row

                    final_df = master_df.apply(reevaluate_con_contexto, axis=1)
                    progress.empty()

                    df_display = final_df.copy()
                    df_display[f"Perf ({periodo})"] = df_display[f"Perf ({periodo})"].apply(fmt_pct)
                    for c in ["P/E Trailing", "P/E Trailing (Prom. Sector)", "P/B", "P/B (Prom. Sector)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(lambda v: fmt_num(v) if pd.notna(v) else "—")
                    for c in ["ROE", "ROE (Prom. Sector)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "—")
                    if "Market Cap" in df_display.columns:
                        df_display["Market Cap"] = df_display["Market Cap"].apply(fmt_bil)
                    for c in ["P/E vs Sector (%)", "P/B vs Sector (%)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "—")
                    if "ROE vs Sector (pp)" in df_display.columns:
                        df_display["ROE vs Sector (pp)"] = df_display["ROE vs Sector (pp)"].apply(lambda x: f"{x*100:.2f} pp" if pd.notna(x) else "—")
                    if "Volatilidad (Anual)" in df_display.columns:
                        df_display["Volatilidad (Anual)"] = df_display["Volatilidad (Anual)"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "—")

                    st.subheader("🏆 Ranking Relativo al Sector")
                    cols_to_show = [
                        "Ticker", "Nombre", "Resultado", "Puntaje", "Sector",
                        "P/E Trailing", "P/E Trailing (Prom. Sector)", "P/E vs Sector (%)",
                        "ROE", "ROE (Prom. Sector)", "ROE vs Sector (pp)",
                        "P/B", "P/B (Prom. Sector)", "P/B vs Sector (%)",
                        "Market Cap", f"Perf ({periodo})", "Volatilidad (Anual)"
                    ]
                    cols_to_show = [c for c in cols_to_show if c in df_display.columns]

                    sort_key = sort_metric if sort_metric in df_display.columns else "Puntaje"
                    tmp_sort = df_display.copy()
                    if sort_key in [f"Perf ({periodo})", "Volatilidad (Anual)", "P/E vs Sector (%)", "P/B vs Sector (%)"]:
                        def to_float_pct(x):
                            if isinstance(x, str) and x.endswith("%"):
                                try: return float(x.replace("%",""))/100.0
                                except: return np.nan
                            return x
                        tmp_sort["_sortcol"] = tmp_sort[sort_key].apply(to_float_pct)
                        sort_key = "_sortcol"

                    df_sorted = tmp_sort.sort_values(by=sort_key, ascending=asc, ignore_index=True).drop(columns=["_sortcol"], errors="ignore")
                    st.dataframe(df_sorted[cols_to_show], use_container_width=True)

                    st.download_button("⬇️ Descargar ranking (CSV)", data=df_sorted[cols_to_show].to_csv(index=False).encode("utf-8"), file_name="ranking_sectorial.csv", mime="text/csv")

                    # Plot top N
                    if len(df_sorted) > 1:
                        top_n = st.slider("Mostrar Top N en gráfico", 1, min(20, len(df_sorted)), min(10, len(df_sorted)))
                        plot_rank = df_sorted.head(top_n)
                        fig_rank = px.bar(plot_rank, x="Ticker", y="Puntaje", color="Sector", title=f"Puntaje por Ticker (Top {top_n})", hover_data=["Nombre", "Resultado", "P/E Trailing", "ROE", "P/B"], template="plotly_dark")
                        st.plotly_chart(fig_rank, use_container_width=True)

                    # Scatter ROE vs P/B
                    if all(col in df_sorted.columns for col in ["ROE", "P/B", "Ticker"]):
                        def to_float_safe(v):
                            if isinstance(v, str) and v.endswith("%"):
                                try: return float(v.replace("%",""))/100.0
                                except: return np.nan
                            try: return float(v)
                            except: return np.nan
                        sc = df_sorted.copy()
                        sc["ROE_num"] = sc["ROE"].apply(to_float_safe)
                        sc["PB_num"] = sc["P/B"].apply(to_float_safe)
                        sc["_MC_num"] = sc["_MarketCapRaw"] if "_MarketCapRaw" in sc.columns else np.nan
                        sc = sc.dropna(subset=["ROE_num", "PB_num"])
                        if not sc.empty:
                            fig_scatter = px.scatter(sc, x="PB_num", y="ROE_num", color="Sector", size="_MC_num", hover_name="Ticker", labels={"PB_num":"P/B","ROE_num":"ROE"}, title="Mapa ROE vs P/B (tamaño = Market Cap)", template="plotly_dark")
                            st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.error("No se pudo obtener información válida para ninguno de los tickers ingresados.")

# Footer note
st.caption(
    "Aviso: los datos provienen de Yahoo Finance vía yfinance. Algunos ratios pueden no estar disponibles o variar según el tipo de activo (acción vs ETF). "
    "Este contenido es solo con fines informativos y no constituye recomendación de inversión."
)
