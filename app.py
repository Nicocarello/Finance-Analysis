# app.py
import time
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from streamlit_extras.stylable_container import stylable_container
# --- IA y Noticias ---
import google.generativeai as genai
import feedparser
from datetime import datetime, timedelta
import requests
import urllib.parse

# Configuración de la API de Gemini (usa tu clave)
# Si usás Streamlit Cloud, podés guardarla en st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "TU_API_KEY_AQUI"))

# --------------------------
# Config & helpers
# --------------------------

# Mensaje de bienvenida
st.info(
    "💡 **Consejo:** Probá con tickers como `AAPL`, `MSFT`, `TSLA` o `SPY` "
    "para explorar análisis fundamental, técnico y de noticias con IA."
)


st.set_page_config(page_title="Analizador de Acciones & ETFs", layout="wide")

# --- CSS PERSONALIZADO PARA MEJORAR UI/UX ---
# Inyectamos nuestro CSS personalizado para "efectos"
st.markdown("""
<style>
    /* Fuente moderna y color base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
        background-color: #0E1117;
    }

    /* Colores principales */
    :root {
        --primary: #4A90E2;
        --secondary: #2E333D;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background-color: var(--secondary);
        border: 1px solid #2E333D;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 16px rgba(74,144,226,0.2);
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #B0B0B0;
        padding: 8px 16px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFFFFF;
        background-color: var(--primary);
    }

    /* Inputs y botones */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #3378C4;
    }
</style>
""", unsafe_allow_html=True)
# --- FIN DE CSS PERSONALIZADO ---

def fmt_pct(x, nd=2):
    return f"{x:.{nd}%}" if x is not None and pd.notna(x) else "—"

def fmt_num(x, nd=2):
    return f"{x:.{nd}f}" if x is not None and pd.notna(x) else "—"

def fmt_bil(x):
    if x is None or pd.isna(x):
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

def calculate_period_performance(hist_df: pd.DataFrame) -> float | None:
    """Calcula el rendimiento porcentual para un DataFrame de historial de precios."""
    if hist_df.empty or "Close" not in hist_df.columns or len(hist_df) < 2:
        return None
    first_close = float(hist_df["Close"].iloc[0])
    last_close = float(hist_df["Close"].iloc[-1])
    if first_close > 0:
        return (last_close / first_close) - 1.0
    return None

def risk_metrics(close: pd.Series, rf_annual: float = 0.0) -> dict:
    """Métricas de riesgo clave sobre precios ajustados (auto_adjust=True)."""
    if close is None or close.size < 2:
        return {}
    ret = close.pct_change().dropna()
    if ret.empty:
        return {}
    ann_factor = 252
    mu = ret.mean() * ann_factor
    sigma = ret.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = (mu - rf_annual) / sigma if sigma and np.isfinite(sigma) else np.nan
    # Drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1.0)
    max_dd = dd.min()
    calmar = mu / abs(max_dd) if max_dd and max_dd < 0 else np.nan
    return {
        "Retorno Anualizado": mu,
        "Volatilidad Anual": sigma,
        "Sharpe": sharpe,
        "Máx. Drawdown": max_dd,
        "Calmar": calmar,
    }

def compute_beta(asset_close: pd.Series, bench_close: pd.Series) -> float | None:
    ret_a = asset_close.pct_change().dropna()
    ret_b = bench_close.pct_change().dropna()
    df = pd.concat([ret_a, ret_b], axis=1).dropna()
    if df.empty:
        return None
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var_b = np.var(df.iloc[:, 1])
    return cov / var_b if var_b else None

def winsorize_series(s: pd.Series, low=0.01, high=0.99):
    if s is None or s.empty:
        return s
    ql, qh = s.quantile([low, high])
    return s.clip(lower=ql, upper=qh)

# --------------------------
# Descarga robusta (reintentos) y batch
# --------------------------
def _retry(fn, *args, retries=3, delay=0.8, **kwargs):
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
        return hist.dropna(subset=["Close"])
    return pd.DataFrame()

def download_history_batch(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = _retry(
        yf.download, tickers=tickers, period=period, interval=interval,
        progress=False, auto_adjust=True, group_by='ticker', threads=True
    )
    out = {}
    if isinstance(raw, pd.DataFrame) and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        # Formato multi-ticker
        first_level = raw.columns.get_level_values(0)
        for tk in tickers:
            if tk in first_level:
                df = raw[tk].copy()
                df = df.dropna(subset=["Close"])
                out[tk] = df
            else:
                out[tk] = pd.DataFrame()
    else:
        # Fallback: quizá fue un solo ticker o respuesta plana
        for tk in tickers:
            try:
                out[tk] = download_history_single(tk, period, interval)
            except Exception:
                out[tk] = pd.DataFrame()
    return out

def capm_alpha(asset_close, bench_close):
    r_a = asset_close.pct_change().dropna()
    r_b = bench_close.pct_change().dropna()
    df = pd.concat([r_a, r_b], axis=1).dropna()
    if df.empty: return None, None, None
    X = sm.add_constant(df.iloc[:,1])
    y = df.iloc[:,0]
    model = sm.OLS(y, X).fit()
    alpha = model.params.get("const")
    beta  = model.params.iloc[1]
    r2    = model.rsquared
    return alpha, beta, r2

# --------------------------
# Funciones de obtención de datos con caching
# --------------------------
@st.cache_resource(show_spinner=False)
def get_ticker_obj(ticker: str):
    # Objetos no serializables -> cache_resource
    return yf.Ticker(ticker)

@st.cache_data(show_spinner=False, ttl=1800)
def get_ticker_info(ticker: str) -> dict:
    """
    Obtiene la información del ticker combinando get_info(), fast_info y fallback manual.
    """
    t = get_ticker_obj(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # Intentar complementar con fast_info
    try:
        fi = t.fast_info or {}
        for k, v in fi.items():
            if k not in info or info[k] is None:
                info[k] = v
    except Exception:
        pass

    # Calcular algunos ratios básicos si hay datos de balance disponibles
    try:
        fin = t.financials
        bs = t.balance_sheet
        if not fin.empty and not bs.empty:
            income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else None
            equity = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
            assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else None
            if income and equity:
                info["returnOnEquity"] = float(income / equity)
            if income and assets:
                info["returnOnAssets"] = float(income / assets)
    except Exception:
        pass

    return info


@st.cache_data(show_spinner=False, ttl=900)
def get_history(ticker: str, period: str, interval: str):
    try:
        return download_history_single(ticker, period, interval)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=900)
def get_histories_batch(tickers: list[str], period: str, interval: str):
    try:
        return download_history_batch(tickers, period, interval)
    except Exception:
        return {tk: pd.DataFrame() for tk in tickers}

def is_etf(info: dict) -> bool:
    qt = (info.get("quoteType") or "").lower()
    name = (info.get("shortName") or info.get("longName") or "").lower()
    return "etf" in qt or "etf" in name


# --------------------------
# Noticias recientes y análisis con IA
# --------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_news(query: str, days: int = 7, max_items_per_topic: int = 5):
    """Busca noticias recientes sobre la empresa y el contexto macroeconómico."""
    if not query:
        return []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    encoded_query = urllib.parse.quote_plus(f"{query} stock OR finance OR earnings")

    def fetch_feed(search_query):
        feed_url = (
            f"https://news.google.com/rss/search?q={search_query}+after:{start_date.strftime('%Y-%m-%d')}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(feed_url)
            return [{
                "title": e.title,
                "link": e.link,
                "published": getattr(e, "published", ""),
                "summary": getattr(e, "summary", "")
            } for e in feed.entries[:max_items_per_topic]]
        except Exception:
            return []

    # Noticias sobre la empresa (micro)
    micro_news = fetch_feed(encoded_query)

    # Noticias macroeconómicas generales
    macro_topics = [
        "US Federal Reserve interest rates",
        "inflation CPI United States",
        "global economic outlook OR IMF",
        "oil prices OR energy market",
        "US-China trade agreement OR tariffs",
        "stock market volatility OR investor sentiment",
    ]
    macro_news = []
    for topic in macro_topics:
        macro_news.extend(fetch_feed(urllib.parse.quote_plus(topic)))

    # Combinar, priorizando las micro (empresa)
    all_news = micro_news + macro_news
    seen_titles = set()
    deduped = []
    for n in all_news:
        if n["title"] not in seen_titles:
            deduped.append(n)
            seen_titles.add(n["title"])

    return deduped[: max_items_per_topic * 2]  # limitar cantidad total


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_news_with_gemini(ticker: str, news_list: list[dict]) -> str:
    """Analiza noticias micro y macro con Gemini para inferir tendencia de mercado."""
    if not news_list:
        return "⚪ No se encontraron noticias recientes para analizar."

    content = "\n".join([f"- {n['title']}: {n['summary']}" for n in news_list])

    prompt = f"""
    Eres un analista financiero experto. Se te proporcionan noticias recientes tanto sobre la acción {ticker}
    como sobre la situación macroeconómica global (tasas, inflación, comercio, materias primas, etc.).

    Tu tarea:
    1. Evalúa si el contexto general es favorable o desfavorable para las acciones en general y para {ticker}.
    2. Determina el sentimiento global del mercado (Positivo / Negativo / Neutro).
    3. Explica brevemente por qué, con referencias a los temas mencionados.

    Noticias recientes:
    {content}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error al generar análisis con Gemini: {e}"

# --------------------------
# Lógica de análisis
# --------------------------
def analizar_ticker(ticker: str, pe_thr: float, roe_thr: float, pb_thr: float,
                    w_pe: float, w_roe: float, w_pb: float):
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
    expense_ratio = safe_pick(info, "annualReportExpenseRatio", scale=1.0)  # ETFs

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
        cond = pb < pb_thr
        razones.append(f"P/B = {pb:.2f} → {'Atractivo' if cond else 'Elevado'} (umbral {pb_thr})")
        scores.append(w_pb if cond else 0.0)
    else:
        razones.append("P/B no disponible")

    total_w = w_pe + w_roe + w_pb
    score = (sum(scores) / total_w) if total_w > 0 else 0
    decision = "✅ Barato" if score >= 0.67 else ("🟡 Neutro" if score >= 0.33 else "❌ Caro")

    nota = None
    if is_etf(info):
        nota = "Este ticker parece ser un ETF: algunos ratios (P/E, ROE, P/B) pueden no estar disponibles o no ser comparables."

    meta = {
        "long_name": long_name,
        "currency": currency,
        "exchange": exchange,
        "sector": sector,
        "industry": industry,
        "is_etf": is_etf(info),
        "nota": nota,
        "score": score,
    }

    return df, decision, razones, meta, t

# --------------------------
# UI
# --------------------------
st.title("📈 Analizador de Acciones & ETFs")

with st.sidebar:
    st.header("Parámetros")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker (ej: AAPL, MSFT, SPY)", "AAPL").strip().upper()
    with col2:
        refresh = st.button("🔄")
    
    # Si se presiona el botón, limpiar cache y volver a correr
    if refresh:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Datos actualizados.", icon="♻️")

    st.subheader("Umbrales")
    pe_thr = st.number_input(
        "Umbral P/E (trailing) máximo",
        min_value=1.0, max_value=100.0, value=15.0, step=0.5,
        help="Relación Precio/Beneficio: cuanto más bajo, más barata está la acción según sus ganancias pasadas."
    )
    roe_thr = st.number_input(
        "Umbral ROE mínimo",
        min_value=0.0, max_value=1.0, value=0.15, step=0.01,
        help="Rentabilidad sobre el capital: mide la eficiencia del uso del capital propio."
    )
    pb_thr = st.number_input(
        "Umbral P/B máximo",
        min_value=0.1, max_value=20.0, value=2.0, step=0.1,
        help="Precio/Valor contable: compara el valor de mercado con el valor contable por acción."
    )


    st.subheader("Pesos")
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
    ai_enabled = st.checkbox(
        "🧠 Activar análisis con Gemini",
        value=True,
        help="Obtiene y analiza noticias recientes con IA para inferir la tendencia del mercado del activo."
    )

# --------------------------
# Tabs
# --------------------------
tab1, tab2 = st.tabs(["🔍 Análisis individual", "🏁 Comparar Tickets"])

with tab1:
    if ticker:
        # --- PASO 1: OBTENER DATOS FUNDAMENTALES Y DE HISTORIAL ---
        df_fund, decision, razones, meta, _t = analizar_ticker(ticker, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb)
        hist = get_history(ticker, periodo, intervalo)

        st.subheader(f"🪪 {meta['long_name']} ({ticker})")
        
        # --- PASO 2: CÁLCULOS PARA MÉTRICAS SUPERIORES ---
        period_performance = None
        last_close = None
        max_periodo = None
        min_periodo = None
        last_volume = None
        avg_volume = None
        vol_delta = None
        plot_df = pd.DataFrame()

        if not hist.empty:
            plot_df = hist.copy()
            period_performance = calculate_period_performance(plot_df)
            last_close = plot_df["Close"].iloc[-1]
            max_periodo = plot_df['High'].max() if 'High' in plot_df.columns else None
            min_periodo = plot_df['Low'].min() if 'Low' in plot_df.columns else None
            last_volume = plot_df["Volume"].iloc[-1] if 'Volume' in plot_df.columns else None

            # Volumen SMA50
            vol_window = 50
            if 'Volume' in plot_df.columns and len(plot_df) >= vol_window:
                plot_df["Vol_SMA50"] = plot_df["Volume"].rolling(vol_window).mean()
                avg_volume = plot_df["Vol_SMA50"].iloc[-1]
                if pd.notna(avg_volume) and avg_volume > 0 and last_volume is not None:
                    vol_delta = (last_volume / avg_volume) - 1.0

            # --- PANEL DE MÉTRICAS COMBINADO (ORDENADO) ---
            st.write("---")

            # 1) Precio y variación
            cols_price = st.columns(5)
            # --- FUNCIÓN CARD (versión HTML pura, sin dependencias externas) ---
            def card(title, value, subtitle=None, color="#4A90E2", icon="📈"):
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1B1F24;
                        border-radius:12px;
                        padding:16px;
                        border:1px solid rgba(255,255,255,0.1);
                        text-align:center;
                        margin-bottom:10px;">
                        <div style='font-size:22px'>{icon}</div>
                        <div style='font-weight:600; font-size:16px; color:#ccc'>{title}</div>
                        <div style='font-size:22px; font-weight:700; color:{color}'>{value}</div>
                        <div style='font-size:13px; color:#999'>{subtitle or ""}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            
            # ==========================
            # 1️⃣ Precio y variación
            # ==========================
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                card("Precio Actual", f"{last_close:.2f}", subtitle="Cierre más reciente", icon="💰")
            with col2:
                card(
                    "Cambio",
                    f"{period_performance:.2%}" if period_performance is not None else "—",
                    subtitle=f"en {periodo}",
                    color="limegreen" if (period_performance and period_performance > 0) else "tomato",
                    icon="📊"
                )
            with col3:
                card("Máximo", f"{max_periodo:.2f}" if max_periodo is not None else "—", subtitle="del período", icon="📈")
            with col4:
                card("Mínimo", f"{min_periodo:.2f}" if min_periodo is not None else "—", subtitle="del período", icon="📉")
            with col5:
                card("Volumen", fmt_bil(last_volume), subtitle="último día", icon="📦")
            
            st.write("---")
            
            # ==========================
            # 2️⃣ Meta fundamental
            # ==========================
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                card("Resultado Fundamental", decision, icon="🧮", color="#FFD700")
            with col2:
                card("Puntaje", f"{meta['score']*100:.0f} %", subtitle="Evaluación combinada", icon="⭐")
            with col3:
                card("Market Cap", fmt_bil(get_ticker_info(ticker).get("marketCap")), subtitle="Capitalización", icon="🏦")
            with col4:
                card("Moneda / Exchange", f"{meta['currency']} · {meta['exchange']}", subtitle="", icon="💱")
            
            st.write("---")
            
            # ==========================
            # 3️⃣ Métricas de riesgo
            # ==========================
            if not plot_df.empty:
                rm = risk_metrics(plot_df["Close"])
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    card("Sharpe", fmt_num(rm.get("Sharpe")), subtitle="Ratio de rentabilidad/riesgo", icon="⚙️")
                with col2:
                    card("Calmar", fmt_num(rm.get("Calmar")), subtitle="Rendimiento vs drawdown", icon="🧭")
                with col3:
                    card("Máx. Drawdown", fmt_pct(rm.get("Máx. Drawdown")), subtitle="Caída máxima", icon="📉", color="orange")
                with col4:
                    card("Ret. Anual", fmt_pct(rm.get("Retorno Anualizado")), subtitle="Rentabilidad esperada", icon="📈", color="limegreen")
                with col5:
                    card("Vol. Anual", fmt_pct(rm.get("Volatilidad Anual")), subtitle="Desviación estándar anual", icon="🌪️")
            
            st.write("---")
            
            # ==========================
            # 4️⃣ Sensibilidad al mercado (CAPM)
            # ==========================
            if not plot_df.empty and benchmark:
                bench_hist = get_history(benchmark, periodo, intervalo)
                if not bench_hist.empty:
                    beta = compute_beta(plot_df["Close"], bench_hist["Close"])
                    try:
                        alpha, beta_ols, r2 = capm_alpha(plot_df["Close"], bench_hist["Close"])
                    except Exception:
                        alpha, beta_ols, r2 = None, None, None
            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        card("β vs Benchmark", fmt_num(beta), subtitle=f"Comparado con {benchmark}", icon="📉", color="#4A90E2")
                    with col2:
                        card("α (CAPM)", fmt_pct(alpha), subtitle="Rendimiento extra vs mercado", icon="⚡", color="#FFD700")
                    with col3:
                        card("R² (CAPM)", fmt_num(r2), subtitle="Ajuste del modelo CAPM", icon="📐", color="#B0E0E6")


        # Info ETF / rendimiento periodo
        if meta["is_etf"] and meta.get("nota"):
            st.info(meta["nota"])

        if period_performance is not None:
            if period_performance > 0:
                st.success(f"📈 El precio ha subido un **{period_performance:.2%}** en el período {periodo}")
            elif period_performance < 0:
                st.error(f"📉 El precio ha bajado un **{abs(period_performance):.2%}** en el período {periodo}")
            else:
                st.info(f"➡️ El precio se mantiene sin cambios en el período {periodo}")

        st.subheader("📊 Ratios Fundamentales")

        order_cols = [
            "Market Cap", "Dividend Yield", "Expense Ratio",
            "P/E Trailing", "P/E Forward", "P/B", "EV/EBITDA",
            "ROE", "ROA", "Profit Margins",
            "Moneda", "Exchange", "Sector", "Industria"
        ]

        # Reordenar columnas según orden lógico, ignorando las que no existan
        df_fund = df_fund[[c for c in order_cols if c in df_fund.columns]]
        st.dataframe(df_fund, use_container_width=True)

        # Radar chart con ratios normalizados
        radar_metrics = {
            "ROE": roe or 0,
            "ROA": roa or 0,
            "Profit Margins": profit_margins or 0,
            "Dividend Yield": dividend_yield or 0,
        }
        
        radar_df = pd.Series(radar_metrics)
        if not radar_df.empty:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df.values * 100,
                theta=radar_df.index,
                fill='toself',
                name=ticker
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title=f"🌐 Perfil Financiero de {ticker}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)


        st.subheader("📌 Evaluación Fundamental")
        st.write("**Explicación de métricas:**")
        for r in razones:
            st.write(f"- {r}")

        st.subheader("📉 Análisis Técnico y Evolución del Precio")
        if plot_df.empty:
            st.error("No hay datos históricos disponibles para los parámetros seleccionados.")
        else:
            # SMA precio
            short_window, long_window = 50, 200
            if sma50_on and len(plot_df) >= short_window:
                plot_df["SMA50"] = plot_df["Close"].rolling(short_window).mean()
            if sma200_on and len(plot_df) >= long_window:
                plot_df["SMA200"] = plot_df["Close"].rolling(long_window).mean()

            # Cruces
            last_signal = None
            golden_crosses = []
            death_crosses = []
            if "SMA50" in plot_df.columns and "SMA200" in plot_df.columns:
                prev_sma50 = plot_df["SMA50"].shift(1)
                prev_sma200 = plot_df["SMA200"].shift(1)
                gc_mask = (plot_df["SMA50"] > plot_df["SMA200"]) & (prev_sma50 <= prev_sma200)
                golden_crosses = plot_df.index[gc_mask].tolist()
                dc_mask = (plot_df["SMA50"] < plot_df["SMA200"]) & (prev_sma50 >= prev_sma200)
                death_crosses = plot_df.index[dc_mask].tolist()
                if golden_crosses:
                    last_gc_date = golden_crosses[-1].strftime('%Y-%m-%d')
                    last_signal = ("golden", f"✨ Golden Cross detectado el {last_gc_date}")
                if death_crosses:
                    last_dc_date = death_crosses[-1].strftime('%Y-%m-%d')
                    if last_signal is None or (death_crosses and golden_crosses and death_crosses[-1] > golden_crosses[-1]):
                        last_signal = ("death", f"💀 Death Cross detectado el {last_dc_date}")

            # Línea de tendencia simple
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(-plot_df["Close"], distance=10)
                if len(peaks) >= 2:
                    x_pivots = peaks
                    y_pivots = plot_df["Close"].iloc[peaks].values
                    coeffs = np.polyfit(x_pivots, y_pivots, 1)
                    plot_df["Trendline"] = coeffs[0] * np.arange(len(plot_df)) + coeffs[1]
            except Exception as e:
                st.caption(f"No se pudo generar la línea de tendencia: {e}")

            # Gráfico principal
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                         low=plot_df['Low'], close=plot_df['Close'], name='Precio'), row=1, col=1)
            for col in ["SMA50", "SMA200", "Trendline"]:
                if col in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], name=col,
                                             line=dict(width=2)), row=1, col=1)
            if golden_crosses and "SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=golden_crosses, y=plot_df.loc[golden_crosses]["SMA50"], name='Golden Cross',
                                         mode='markers', marker=dict(color='gold', size=12, symbol='star')), row=1, col=1)
            if death_crosses and "SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=death_crosses, y=plot_df.loc[death_crosses]["SMA50"], name='Death Cross',
                                         mode='markers', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)

            if 'Volume' in plot_df.columns:
                fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Volumen',
                                     marker_color='lightgrey'), row=2, col=1)
            if "Vol_SMA50" in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol_SMA50"], name='Volumen SMA(50)',
                                         line=dict(width=1, color='slategray')), row=2, col=1)

            fig.update_layout(
                title=f"Análisis Técnico - {ticker} | Cambio {periodo}: {f'{period_performance:.2%}' if period_performance is not None else ''}",
                xaxis_rangeslider_visible=False, height=600, legend_title="Series"
            )
            # Ocultar fines de semana/feriados y spikes
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showspikes=True, spikemode="across")
            fig.update_yaxes(title_text=f"Precio ({meta['currency']})", row=1, col=1, type="log" if log_scale else "linear")
            fig.update_yaxes(title_text="Volumen", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # Señales
            if last_signal:
                if last_signal[0] == "golden":
                    st.success(last_signal[1])
                else:
                    st.error(last_signal[1])
            elif sma50_on and sma200_on:
                st.info("No se han detectado cruces de medias móviles (50/200) en el período seleccionado.")

            # RSI opcional (Wilder)
            if rsi_on and len(plot_df) > 14:
                close = plot_df["Close"]
                delta = close.diff()
                roll_up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                roll_down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                rs = roll_up / roll_down
                rsi = 100 - (100 / (1 + rs))
                rsi_df = pd.DataFrame({"RSI": rsi}).dropna()
                if not rsi_df.empty:
                    fig_rsi = px.line(rsi_df, x=rsi_df.index, y="RSI", title=f"RSI (14) - {ticker}")
                    fig_rsi.update_layout(xaxis_title="Fecha", yaxis_title="RSI")
                    fig_rsi.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1)
                    fig_rsi.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1)
                    st.plotly_chart(fig_rsi, use_container_width=True)
        if ai_enabled:
            st.write("---")
            st.subheader("🧠 Análisis AI de Noticias Recientes")

            with st.spinner(f"Buscando y analizando noticias de los últimos 7 días sobre {ticker}..."):
                query_name = meta.get("long_name") or ticker
                news = fetch_recent_news(query_name)
                ai_summary = analyze_news_with_gemini(ticker, news)

            if news:
                with st.expander("🗞️ Ver noticias recientes"):
                    for n in news:
                        st.markdown(f"- [{n['title']}]({n['link']})  \n<sub>{n['published']}</sub>", unsafe_allow_html=True)
            else:
                st.info("No se encontraron noticias recientes.")

            st.markdown(ai_summary)
        

with tab2:
    st.header("📊 Comparador de Tickers")
    st.write("Ingresa una lista de tickers separados por comas. Ej: `AAPL, MSFT, GOOG, JPM, BAC`")
    tickers_input = st.text_area("Tickers", value="AAPL, MSFT, GOOG, JPM, BAC").strip()

    analysis_mode = st.radio(
        "Modo de Análisis",
        ("Contexto Sectorial (Recomendado)", "Umbrales Absolutos (Original)"),
        horizontal=True,
        help="El comparador sectorial los contrasta con sus pares; el modo absoluto usa los umbrales de la barra lateral."
    )
    st.caption("El comparador usa los pesos, período e intervalo definidos en la barra lateral.")

    # --- Limpieza y validación de tickers ---
    if not tickers_input:
        st.info("Ingresa al menos un ticker para comenzar el análisis.")
    else:
        # normalizamos: upper, strip, únicos, sin vacíos
        tickers = []
        for t in tickers_input.split(","):
            t = t.strip().upper()
            if t and t not in tickers:
                tickers.append(t)

        if len(tickers) == 0:
            st.warning("La lista de tickers está vacía.")
        else:
            # --- Controles extra de la vista Comparador ---
            colc1, colc2, colc3 = st.columns([1,1,1])
            with colc1:
                sort_metric = st.selectbox(
                    "Ordenar por",
                    ["Puntaje", f"Perf ({periodo})", "Volatilidad (Anual)", "Market Cap", "ROE", "P/E Trailing", "P/B"],
                    index=0
                )
            with colc2:
                asc = st.toggle("Ascendente", value=False, help="Desactiva para ver los mejores arriba")
            with colc3:
                mcap_min = st.number_input("Market Cap mínimo (USD, millones)", min_value=0, value=0, step=100)
                mcap_min = mcap_min * 1_000_000  # a USD

            # ============================
            # MODO: UMBRALES ABSOLUTOS
            # ============================
            if analysis_mode == "Umbrales Absolutos (Original)":
                progress = st.progress(0.0, text="Analizando con umbrales absolutos…")
                rows = []
                failed = []

                for i, tk in enumerate(tickers, start=1):
                    try:
                        df_i, decision_i, _, meta_i, _ = analizar_ticker(
                            tk, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb
                        )
                        hist_i = get_history(tk, periodo, intervalo)
                        perf = calculate_period_performance(hist_i)
                        volatilidad = None
                        if not hist_i.empty:
                            retornos = hist_i['Close'].pct_change()
                            volatilidad = retornos.std(ddof=0) * np.sqrt(252)

                        # market cap crudo para filtro/sort
                        try:
                            raw_mcap = get_ticker_info(tk).get("marketCap")
                        except Exception:
                            raw_mcap = None

                        rows.append({
                            "Ticker": tk,
                            "Nombre": meta_i["long_name"],
                            "Resultado": decision_i,
                            "Puntaje": round(meta_i["score"] * 100, 1),
                            "P/E Trailing": df_i.get("P/E Trailing", ["—"])[0] if isinstance(df_i.get("P/E Trailing"), pd.Series) else df_i.get("P/E Trailing", ["—"])[0] if "P/E Trailing" in df_i else "—",
                            "ROE": df_i.get("ROE", ["—"])[0] if "ROE" in df_i else "—",
                            "P/B": df_i.get("P/B", ["—"])[0] if "P/B" in df_i else "—",
                            "Market Cap": df_i.get("Market Cap", ["—"])[0] if "Market Cap" in df_i else "—",
                            "_MarketCapRaw": raw_mcap,
                            "Sector": meta_i["sector"],
                            f"Perf ({periodo})": fmt_pct(perf, 2),
                            "Volatilidad (Anual)": fmt_pct(volatilidad, 2),
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

                    # Filtro por Market Cap mínimo (usa crudo)
                    if mcap_min > 0 and "_MarketCapRaw" in rank_df:
                        rank_df = rank_df[(rank_df["_MarketCapRaw"].fillna(0) >= mcap_min)]

                    # Orden dinámico
                    sort_key = sort_metric
                    if sort_metric in [f"Perf ({periodo})", "Volatilidad (Anual)"]:
                        # convertir a float si vienen como strings "12.34%"
                        def to_float_pct(x):
                            if isinstance(x, str) and x.endswith("%"):
                                try: return float(x.replace("%",""))/100.0
                                except: return np.nan
                            return x
                        rank_df["_sortcol"] = rank_df[sort_metric].apply(to_float_pct)
                        sort_key = "_sortcol"
                    elif sort_metric == "Market Cap":
                        # usar crudo si existe
                        sort_key = "_MarketCapRaw" if "_MarketCapRaw" in rank_df.columns else "Market Cap"

                    rank_df_sorted = rank_df.sort_values(by=sort_key, ascending=asc, ignore_index=True).drop(columns=["_sortcol"], errors="ignore")

                    st.subheader("🏆 Ranking (Umbrales Absolutos)")
                    st.dataframe(rank_df_sorted.drop(columns=["_MarketCapRaw"], errors="ignore"), use_container_width=True)

                    # Export CSV
                    st.download_button(
                        "⬇️ Descargar ranking (CSV)",
                        data=rank_df_sorted.drop(columns=["_MarketCapRaw"], errors="ignore").to_csv(index=False).encode("utf-8"),
                        file_name="ranking_absoluto.csv",
                        mime="text/csv"
                    )

            # ============================
            # MODO: CONTEXTO SECTORIAL
            # ============================
            elif analysis_mode == "Contexto Sectorial (Recomendado)":
                # FASE 1: datos en batch
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
                master_df = master_df[master_df["Sector"] != "Error"]

                # Filtros previos
                if mcap_min > 0 and "_MarketCapRaw" in master_df.columns:
                    master_df = master_df[(master_df["_MarketCapRaw"].fillna(0) >= mcap_min)]

                if not master_df.empty:
                    # Selector de sectores (opcional)
                    sectores = sorted([s for s in master_df["Sector"].dropna().unique() if s])
                    sel_sectores = st.multiselect("Filtrar sectores", options=sectores, default=sectores)
                    master_df = master_df[master_df["Sector"].isin(sel_sectores)] if sel_sectores else master_df

                    # FASE 2: promedios sectoriales robustos (mediana + winsor)
                    progress = st.progress(1.0, text="Fase 2/2: Analizando contexto sectorial…")
                    metrics_to_average = ["P/E Trailing", "ROE", "P/B"]

                    def robust_sector_stat(x: pd.Series):
                        x_w = winsorize_series(x.dropna())
                        return x_w.median()

                    for metric in metrics_to_average:
                        master_df[f"{metric} (Prom. Sector)"] = (
                            master_df.groupby('Sector')[metric]
                            .transform(robust_sector_stat)
                        )

                    # Reevaluación con comparación a mediana de sector
                    def reevaluar_con_contexto(row):
                        scores = []
                        # P/E < sector (mejor más bajo)
                        pe, pe_avg = row.get("P/E Trailing"), row.get("P/E Trailing (Prom. Sector)")
                        if pd.notna(pe) and pd.notna(pe_avg) and pe_avg > 0:
                            scores.append(w_pe if pe < pe_avg else 0)
                            row["P/E vs Sector (%)"] = (pe / pe_avg - 1.0) if pe_avg else np.nan
                        else:
                            row["P/E vs Sector (%)"] = np.nan

                        # ROE > sector (mejor más alto)
                        roe, roe_avg = row.get("ROE"), row.get("ROE (Prom. Sector)")
                        if pd.notna(roe) and pd.notna(roe_avg):
                            scores.append(w_roe if roe > roe_avg else 0)
                            row["ROE vs Sector (pp)"] = (roe - roe_avg) if pd.notna(roe_avg) else np.nan
                        else:
                            row["ROE vs Sector (pp)"] = np.nan

                        # P/B < sector
                        pb, pb_avg = row.get("P/B"), row.get("P/B (Prom. Sector)")
                        if pd.notna(pb) and pd.notna(pb_avg) and pb_avg > 0:
                            scores.append(w_pb if pb < pb_avg else 0)
                            row["P/B vs Sector (%)"] = (pb / pb_avg - 1.0) if pb_avg else np.nan
                        else:
                            row["P/B vs Sector (%)"] = np.nan

                        total_w = w_pe + w_roe + w_pb
                        score = (sum(scores) / total_w) if total_w > 0 else 0
                        decision = "✅ Atractivo vs Pares" if score >= 0.67 else ("🟡 Neutro vs Pares" if score >= 0.33 else "❌ Caro vs Pares")
                        row["Puntaje"] = round(score * 100, 1)
                        row["Resultado"] = decision
                        return row

                    final_df = master_df.apply(reevaluar_con_contexto, axis=1)
                    progress.empty()

                    # Formateo visual
                    df_display = final_df.copy()
                    df_display[f"Perf ({periodo})"] = df_display[f"Perf ({periodo})"].apply(fmt_pct)
                    for c in ["P/E Trailing", "P/E Trailing (Prom. Sector)", "P/B", "P/B (Prom. Sector)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(fmt_num)
                    for c in ["ROE", "ROE (Prom. Sector)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(fmt_pct)
                    if "Market Cap" in df_display.columns:
                        df_display["Market Cap"] = df_display["Market Cap"].apply(fmt_bil)
                    # deltas relativos
                    for c in ["P/E vs Sector (%)", "P/B vs Sector (%)"]:
                        if c in df_display.columns: df_display[c] = df_display[c].apply(fmt_pct)
                    if "ROE vs Sector (pp)" in df_display.columns:
                        df_display["ROE vs Sector (pp)"] = df_display["ROE vs Sector (pp)"].apply(lambda x: f"{x*100:.2f} pp" if pd.notna(x) else "—")
                    if "Volatilidad (Anual)" in df_display.columns:
                        df_display["Volatilidad (Anual)"] = df_display["Volatilidad (Anual)"].apply(fmt_pct)

                    st.subheader("🏆 Ranking Relativo al Sector")
                    cols_to_show = [
                        "Ticker", "Nombre", "Resultado", "Puntaje", "Sector",
                        "P/E Trailing", "P/E Trailing (Prom. Sector)", "P/E vs Sector (%)",
                        "ROE", "ROE (Prom. Sector)", "ROE vs Sector (pp)",
                        "P/B", "P/B (Prom. Sector)", "P/B vs Sector (%)",
                        "Market Cap", f"Perf ({periodo})", "Volatilidad (Anual)"
                    ]
                    cols_to_show = [c for c in cols_to_show if c in df_display.columns]
                    # Orden dinámico
                    sort_key = sort_metric if sort_metric in df_display.columns else "Puntaje"
                    tmp_sort = df_display.copy()
                    if sort_key in [f"Perf ({periodo})", "Volatilidad (Anual)", "P/E vs Sector (%)", "P/B vs Sector (%)"]:
                        # crear columna numérica temporal para ordenar
                        def to_float_pct(x):
                            if isinstance(x, str) and x.endswith("%"):
                                try: return float(x.replace("%",""))/100.0
                                except: return np.nan
                            return x
                        tmp_sort["_sortcol"] = tmp_sort[sort_key].apply(to_float_pct)
                        sort_key = "_sortcol"

                    df_sorted = tmp_sort.sort_values(by=sort_key, ascending=asc, ignore_index=True).drop(columns=["_sortcol"], errors="ignore")
                    st.dataframe(df_sorted[cols_to_show], use_container_width=True)

                    # Export CSV
                    st.download_button(
                        "⬇️ Descargar ranking (CSV)",
                        data=df_sorted[cols_to_show].to_csv(index=False).encode("utf-8"),
                        file_name="ranking_sectorial.csv",
                        mime="text/csv"
                    )

                    # Gráfico de ranking (Top N)
                    if len(df_sorted) > 1:
                        top_n = st.slider("Mostrar Top N en gráfico", 1, min(20, len(df_sorted)), min(10, len(df_sorted)))
                        plot_rank = df_sorted.head(top_n)
                        fig_rank = px.bar(
                            plot_rank, x="Ticker", y="Puntaje", color="Sector",
                            title=f"Puntaje por Ticker (Top {top_n})",
                            hover_data=["Nombre", "Resultado", "P/E Trailing", "ROE", "P/B"]
                        )
                        st.plotly_chart(fig_rank, use_container_width=True)

                    # Scatter ROE vs P/B (tamaño=Market Cap)
                    if all(col in df_sorted.columns for col in ["ROE", "P/B", "Ticker"]):
                        # Convertir ROE/PB a valores numéricos para el scatter
                        def to_float_safe(v):
                            if isinstance(v, str) and v.endswith("%"):
                                try: return float(v.replace("%",""))/100.0
                                except: return np.nan
                            try: return float(v)
                            except: return np.nan

                        sc = df_sorted.copy()
                        sc["ROE_num"] = sc["ROE"].apply(to_float_safe)
                        sc["PB_num"] = sc["P/B"].apply(to_float_safe)
                        sc["MC_num"] = sc["_MarketCapRaw"] if "_MarketCapRaw" in sc.columns else np.nan

                        sc = sc.dropna(subset=["ROE_num", "PB_num"])
                        if not sc.empty:
                            fig_scatter = px.scatter(
                                sc, x="PB_num", y="ROE_num", color="Sector", size="MC_num",
                                hover_name="Ticker",
                                labels={"PB_num": "P/B", "ROE_num": "ROE"},
                                title="Mapa ROE vs P/B (tamaño = Market Cap)"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.error("No se pudo obtener información válida para ninguno de los tickers ingresados.")

# Footer
st.caption(
    "Aviso: los datos provienen de Yahoo Finance vía yfinance. Algunos ratios pueden no estar disponibles o variar según el tipo de activo (acción vs ETF). "
    "Este contenido es solo con fines informativos y no constituye recomendación de inversión."
)
