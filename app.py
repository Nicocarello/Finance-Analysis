# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

# --------------------------
# Config & helpers
# --------------------------
st.set_page_config(page_title="Analizador de Acciones & ETFs", layout="wide")

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

# --------------------------
# Funciones de obtención de datos con caching
@st.cache_resource(show_spinner=False)
def get_ticker_obj(ticker: str):
    # Objetos no serializables -> cache_resource
    return yf.Ticker(ticker)

@st.cache_data(show_spinner=False, ttl=1800)
def get_ticker_info(ticker: str) -> dict:
    # Datos serializables -> cache_data
    t = get_ticker_obj(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        try:
            fi = t.fast_info or {}
            info = dict(fi)
        except Exception:
            info = {}
    return info

@st.cache_data(show_spinner=False, ttl=900)
def get_history(ticker: str, period: str, interval: str):
    try:
        hist = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0] for c in hist.columns]
        hist = hist.dropna(subset=["Close"])
        return hist
    except Exception:
        return pd.DataFrame()

def is_etf(info: dict) -> bool:
    qt = (info.get("quoteType") or "").lower()
    name = (info.get("shortName") or info.get("longName") or "").lower()
    return "etf" in qt or "etf" in name

# --------------------------
# Lógica de análisis
# --------------------------
def analizar_ticker(ticker: str, pe_thr: float, roe_thr: float, pb_thr: float, w_pe: float, w_roe: float, w_pb: float):
    t = get_ticker_obj(ticker)
    info = get_ticker_info(ticker)

    def pick(*keys, default=None, scale=None):
        for k in keys:
            v = info.get(k)
            if v is not None and pd.notna(v):
                return v * scale if (scale is not None and v is not None) else v
        return default

    market_cap = pick("marketCap", "market_cap")
    trailing_pe = pick("trailingPE", "trailing_pe")
    forward_pe = pick("forwardPE", "forward_pe")
    pb = pick("priceToBook", "price_to_book")
    ev_ebitda = pick("enterpriseToEbitda", "ev_to_ebitda")
    roe = pick("returnOnEquity", "return_on_equity", scale=1.0)
    roa = pick("returnOnAssets", "return_on_assets", scale=1.0)
    profit_margins = pick("profitMargins", "profit_margins", scale=1.0)
    dividend_yield = pick("dividendYield", "dividend_yield", scale=1.0)

    currency = pick("currency") or pick("financialCurrency") or "—"
    exchange = pick("exchange") or "—"
    sector = pick("sector") or "—"
    industry = pick("industry") or "—"
    long_name = pick("longName") or pick("shortName") or ticker

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
    ticker = st.text_input("Ticker (ej: AAPL, MSFT, SPY)", "AAPL").strip().upper()

    st.subheader("Umbrales")
    pe_thr = st.number_input("Umbral P/E (trailing) máximo", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
    roe_thr = st.number_input("Umbral ROE mínimo", min_value=0.0, max_value=1.0, value=0.15, step=0.01, help="En fracción (0.15 = 15%)")
    pb_thr = st.number_input("Umbral P/B máximo", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    st.subheader("Pesos (importancia)")
    w_pe = st.slider("Peso P/E", 0.0, 1.0, 0.34)
    w_roe = st.slider("Peso ROE", 0.0, 1.0, 0.33)
    w_pb = st.slider("Peso P/B", 0.0, 1.0, 0.33)

    st.subheader("Serie de precios")
    periodo = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=2)
    intervalo = st.selectbox("Intervalo", ["1d", "5d", "1wk", "1mo", "3mo"], index=0)
    sma50_on = st.checkbox("Mostrar SMA 50", value=True)
    sma200_on = st.checkbox("Mostrar SMA 200", value=False)
    rsi_on = st.checkbox("Mostrar RSI (panel aparte)", value=False)
    log_scale = st.checkbox("Escala logarítmica", value=False)

# --------------------------
# Tabs
# --------------------------
tab1, tab2 = st.tabs(["🔍 Análisis individual", "🏁 Comparar Tickets"])

with tab1:
    if ticker:
        # --- PASO 1: OBTENER DATOS FUNDAMENTALES Y DE HISTORIAL ---
        df, decision, razones, meta, _t = analizar_ticker(ticker, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb)
        hist = get_history(ticker, periodo, intervalo)

        st.subheader(f"🪪 {meta['long_name']} ({ticker})")
        
        # --- PASO 2: CALCULAR TEMPRANAMENTE LAS MÉTRICAS TÉCNICAS Y DE RENDIMIENTO ---
        # Inicializamos todas las variables que usaremos en las métricas para evitar errores
        period_performance = None
        last_close = None
        max_periodo = None
        min_periodo = None
        last_volume = None
        avg_volume = None
        vol_delta = None
        plot_df = pd.DataFrame() # DataFrame vacío por si 'hist' falla

        if not hist.empty:
            plot_df = hist.copy() # Usaremos plot_df consistentemente
            
            # Cálculo de rendimiento
            period_performance = calculate_period_performance(plot_df) # Usando la función helper

            # Cálculo de métricas técnicas para el panel superior
            last_close = plot_df["Close"].iloc[-1]
            max_periodo = plot_df['High'].max()
            min_periodo = plot_df['Low'].min()
            last_volume = plot_df["Volume"].iloc[-1]
            
            # Calcular SMA de volumen para el delta
            vol_window = 50
            if len(plot_df) >= vol_window:
                plot_df["Vol_SMA50"] = plot_df["Volume"].rolling(vol_window).mean()
                avg_volume = plot_df["Vol_SMA50"].iloc[-1]
                if pd.notna(avg_volume) and avg_volume > 0:
                    vol_delta = (last_volume / avg_volume) - 1.0

        # --- PASO 3: MOSTRAR EL PANEL DE MÉTRICAS COMBINADO ---
        st.write("---") # Separador visual
        
        # Fila 1: Métricas Fundamentales
        cols1 = st.columns(5)
        cols1[0].metric("Resultado Fundamental", decision)
        cols1[1].metric("Puntaje", f"{meta['score']*100:.0f} %")
        cols1[2].metric(f"Cambio {periodo}",
                       f"{period_performance:.2%}" if period_performance is not None else "—",
                       delta=f"{period_performance:.2%}" if period_performance is not None else None)
        cols1[3].metric("Moneda", meta["currency"])
        cols1[4].metric("Exchange", meta["exchange"])

        # Fila 2: Métricas de Mercado y Técnicas
        cols2 = st.columns(4)
        cols2[0].metric("Precio Actual", f"{last_close:.2f}" if last_close is not None else "—")
        cols2[1].metric("Máximo del Período", f"{max_periodo:.2f}" if max_periodo is not None else "—")
        cols2[2].metric("Mínimo del Período", f"{min_periodo:.2f}" if min_periodo is not None else "—")
        cols2[3].metric("Volumen Hoy", f"{fmt_bil(last_volume)}" if last_volume is not None else "—", 
                      delta=f"{vol_delta:.1%}" if vol_delta is not None else None,
                      help="Comparado con el promedio de 50 días.")
        st.write("---")

        # --- El resto de la UI sigue como antes ---
        if meta["is_etf"]:
            st.info(meta["nota"])

        if period_performance is not None:
            if period_performance > 0:
                st.success(f"📈 El precio ha subido un **{period_performance:.2%}** en el período {periodo}")
            elif period_performance < 0:
                st.error(f"📉 El precio ha bajado un **{abs(period_performance):.2%}** en el período {periodo}")
            else:
                st.info(f"➡️ El precio se mantiene sin cambios en el período {periodo}")

        st.subheader("📊 Ratios Fundamentales")
        st.dataframe(df, use_container_width=True)

        st.subheader("📌 Evaluación Fundamental")
        st.write("**Explicación de métricas:**")
        for r in razones:
            st.write(f"- {r}")

        st.subheader("📉 Análisis Técnico y Evolución del Precio")
        if plot_df.empty:
            st.error("No hay datos históricos disponibles para los parámetros seleccionados.")
        else:
            # Ahora el código del gráfico técnico funcionará sin problemas,
            # pero ya no necesita mostrar las métricas al final porque ya están arriba.
            
            # --- CÁLCULO DE INDICADORES (SMA de precio, etc.) ---
            short_window, long_window = 50, 200
            if sma50_on and len(plot_df) >= short_window:
                plot_df["SMA50"] = plot_df["Close"].rolling(short_window).mean()
            if sma200_on and len(plot_df) >= long_window:
                plot_df["SMA200"] = plot_df["Close"].rolling(long_window).mean()

            # --- DETECCIÓN DE SEÑALES Y LÍNEA DE TENDENCIA ---
            # (Todo este bloque de código se mantiene exactamente igual que antes)
            last_signal = None
            golden_crosses = []
            death_crosses = []
            if "SMA50" in plot_df and "SMA200" in plot_df:
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
                    if last_signal is None or death_crosses[-1] > golden_crosses[-1]:
                        last_signal = ("death", f"💀 Death Cross detectado el {last_dc_date}")
            
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

            # --- CREACIÓN DEL GRÁFICO (sin cambios) ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                       low=plot_df['Low'], close=plot_df['Close'], name='Precio'), row=1, col=1)
            for col in ["SMA50", "SMA200", "Trendline"]:
                if col in plot_df:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], name=col,
                                             line=dict(width=2)), row=1, col=1)
            if golden_crosses:
                fig.add_trace(go.Scatter(x=golden_crosses, y=plot_df.loc[golden_crosses]["SMA50"], name='Golden Cross', mode='markers',
                                         marker=dict(color='gold', size=12, symbol='star')), row=1, col=1)
            if death_crosses:
                fig.add_trace(go.Scatter(x=death_crosses, y=plot_df.loc[death_crosses]["SMA50"], name='Death Cross', mode='markers',
                                         marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Volumen',
                                 marker_color='lightgrey'), row=2, col=1)
            if "Vol_SMA50" in plot_df:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol_SMA50"], name='Volumen SMA(50)',
                                         line=dict(width=1, color='slategray')), row=2, col=1)

            fig.update_layout(title=f"Análisis Técnico - {ticker} | Cambio {periodo}: {f'{period_performance:.2%}' if period_performance is not None else ''}",
                              xaxis_rangeslider_visible=False, height=600, legend_title="Series")
            fig.update_yaxes(title_text=f"Precio ({meta['currency']})", row=1, col=1, type="log" if log_scale else "linear")
            fig.update_yaxes(title_text="Volumen", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # --- SEÑALES DE CRUCE (sin el bloque de métricas que ya movimos) ---
            if last_signal:
                if last_signal[0] == "golden":
                    st.success(last_signal[1])
                else:
                    st.error(last_signal[1])
            elif sma50_on and sma200_on:
                st.info("No se han detectado cruces de medias móviles (50/200) en el período seleccionado.")
            
            # --- CÁLCULO Y GRÁFICO DE RSI (sin cambios) ---
            if rsi_on and len(plot_df) > 14:
                close = plot_df["Close"]
                delta = close.diff()
                up = delta.clip(lower=0.0)
                down = -1 * delta.clip(upper=0.0)
                roll_up = up.ewm(com=13, adjust=False).mean()
                roll_down = down.ewm(com=13, adjust=False).mean()
                rs = roll_up / roll_down
                rsi = 100 - (100 / (1 + rs))
                rsi_df = pd.DataFrame({"RSI": rsi}).dropna()
                if not rsi_df.empty:
                    fig_rsi = px.line(rsi_df, x=rsi_df.index, y="RSI", title=f"RSI (14) - {ticker}")
                    fig_rsi.update_layout(xaxis_title="Fecha", yaxis_title="RSI")
                    fig_rsi.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1)
                    fig_rsi.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1)
                    st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    st.header("📊 Comparador de Tickers")
    st.write("Ingresa una lista de tickers separados por comas. Ej: `AAPL, MSFT, GOOG, JPM, BAC`")
    tickers_input = st.text_area("Tickers", value="AAPL, MSFT, GOOG, JPM, BAC").strip()

    analysis_mode = st.radio(
        "Modo de Análisis",
        ("Contexto Sectorial (Recomendado)", "Umbrales Absolutos (Original)"),
        horizontal=True,
        help="Elige cómo evaluar los tickers. El contexto sectorial los compara con sus pares, mientras que los umbrales absolutos usan los valores fijos de la barra lateral."
    )
    st.caption("El comparador siempre utiliza los pesos, período e intervalo definidos en la barra lateral.")

    if not tickers_input:
        st.info("Ingresa al menos un ticker para comenzar el análisis.")
    else:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) == 0:
            st.warning("La lista de tickers está vacía.")
        else:
            # --- LÓGICA DE ANÁLISIS ABSOLUTO (MÉTODO ORIGINAL) ---
            if analysis_mode == "Umbrales Absolutos (Original)":
                progress = st.progress(0.0, text="Analizando con umbrales absolutos…")
                rows = []
                for i, tk in enumerate(tickers, start=1):
                    try:
                        # Reutilizamos la función de análisis original
                        df_i, decision_i, _, meta_i, _ = analizar_ticker(
                            tk, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb
                        )
                        hist_i = get_history(tk, periodo, intervalo)
                        perf = calculate_period_performance(hist_i) # Usando la función helper que sugerí
                        
                        volatilidad = None
                        if not hist_i.empty:
                            retornos = hist_i['Close'].pct_change()
                            volatilidad = retornos.std() * np.sqrt(252)

                        rows.append({
                            "Ticker": tk,
                            "Nombre": meta_i["long_name"],
                            "Resultado": decision_i,
                            "Puntaje": round(meta_i["score"] * 100, 1),
                            "P/E Trailing": df_i["P/E Trailing"].iloc[0],
                            "ROE": df_i["ROE"].iloc[0],
                            "P/B": df_i["P/B"].iloc[0],
                            "Market Cap": df_i["Market Cap"].iloc[0],
                            "Sector": meta_i["sector"],
                            f"Perf ({periodo})": fmt_pct(perf, 2),
                            "Volatilidad (Anual)": fmt_pct(volatilidad, 2),
                        })
                    except Exception as e:
                        st.toast(f"Error analizando {tk}: {e}", icon="⚠️")
                        rows.append({"Ticker": tk, "Nombre": "Error", "Puntaje": 0})
                    finally:
                        progress.progress(i / len(tickers), text=f"Analizando {tk} ({i}/{len(tickers)})")
                
                progress.empty()
                if rows:
                    rank_df_sorted = pd.DataFrame(rows).sort_values(by="Puntaje", ascending=False, ignore_index=True)
                    st.subheader("🏆 Ranking (Umbrales Absolutos)")
                    st.dataframe(rank_df_sorted, use_container_width=True)
                    # Aquí puedes añadir los gráficos y filtros que ya tenías
            
            # --- LÓGICA DE ANÁLISIS SECTORIAL (NUEVO MÉTODO) ---
            elif analysis_mode == "Contexto Sectorial (Recomendado)":
                # FASE 1: Recopilar todos los datos numéricos primero
                progress = st.progress(0.0, text="Fase 1/2: Recopilando datos...")
                all_data = []
                for i, tk in enumerate(tickers, start=1):
                    try:
                        info = get_ticker_info(tk)
                        hist = get_history(tk, periodo, intervalo)
                        
                        perf = calculate_period_performance(hist)
                        volatilidad = None
                        if not hist.empty:
                            volatilidad = hist['Close'].pct_change().std() * np.sqrt(252)

                        all_data.append({
                            "Ticker": tk,
                            "Nombre": info.get("longName") or info.get("shortName") or tk,
                            "Sector": info.get("sector") or "Desconocido",
                            "P/E Trailing": info.get("trailingPE"),
                            "ROE": info.get("returnOnEquity"),
                            "P/B": info.get("priceToBook"),
                            "Market Cap": info.get("marketCap"),
                            f"Perf ({periodo})": perf,
                            "Volatilidad (Anual)": volatilidad,
                        })
                    except Exception as e:
                        st.toast(f"No se pudieron obtener datos para {tk}: {e}", icon="⚠️")
                        all_data.append({"Ticker": tk, "Sector": "Error"})
                    finally:
                        progress.progress(i / len(tickers), text=f"Fase 1/2: Recopilando {tk}")
                
                master_df = pd.DataFrame(all_data).dropna(subset=['Sector']).reset_index(drop=True)
                master_df = master_df[master_df["Sector"] != "Error"]

                if not master_df.empty:
                    # FASE 2: Calcular promedios sectoriales y re-evaluar
                    progress.progress(1.0, text="Fase 2/2: Analizando contexto sectorial...")
                    
                    metrics_to_average = ["P/E Trailing", "ROE", "P/B"]
                    for metric in metrics_to_average:
                        # Usamos transform para crear una nueva columna con el promedio del sector de cada fila
                        master_df[f"{metric} (Prom. Sector)"] = master_df.groupby('Sector')[metric].transform('mean')

                    def reevaluar_con_contexto(row):
                        scores, razones = [], []
                        
                        # P/E vs Promedio del Sector
                        pe, pe_avg = row["P/E Trailing"], row["P/E Trailing (Prom. Sector)"]
                        if pd.notna(pe) and pd.notna(pe_avg) and pe_avg > 0:
                            cond = pe < pe_avg
                            scores.append(w_pe if cond else 0)
                        
                        # ROE vs Promedio del Sector
                        roe, roe_avg = row["ROE"], row["ROE (Prom. Sector)"]
                        if pd.notna(roe) and pd.notna(roe_avg):
                            cond = roe > roe_avg
                            scores.append(w_roe if cond else 0)
                        
                        # P/B vs Promedio del Sector
                        pb, pb_avg = row["P/B"], row["P/B (Prom. Sector)"]
                        if pd.notna(pb) and pd.notna(pb_avg) and pb_avg > 0:
                            cond = pb < pb_avg
                            scores.append(w_pb if cond else 0)
                        
                        total_w = w_pe + w_roe + w_pb
                        score = (sum(scores) / total_w) if total_w > 0 else 0
                        
                        decision = "✅ Atractivo vs Pares" if score >= 0.67 else ("🟡 Neutro vs Pares" if score >= 0.33 else "❌ Caro vs Pares")
                        
                        row["Puntaje"] = round(score * 100, 1)
                        row["Resultado"] = decision
                        return row

                    final_df = master_df.apply(reevaluar_con_contexto, axis=1)
                    progress.empty()
                    
                    # Formatear y mostrar tabla final
                    st.subheader("🏆 Ranking Relativo al Sector")
                    
                    df_display = final_df.sort_values("Puntaje", ascending=False, ignore_index=True)
                    
                    # Formateo para visualización
                    format_mapping = {
                        "P/E Trailing": fmt_num, "P/E Trailing (Prom. Sector)": fmt_num,
                        "ROE": fmt_pct, "ROE (Prom. Sector)": fmt_pct,
                        "P/B": fmt_num, "P/B (Prom. Sector)": fmt_num,
                        "Market Cap": fmt_bil,
                        f"Perf ({periodo})": fmt_pct, "Volatilidad (Anual)": fmt_pct,
                    }
                    for col, func in format_mapping.items():
                        if col in df_display:
                            df_display[col] = df_display[col].apply(func)
                    
                    cols_to_show = [
                        "Ticker", "Nombre", "Resultado", "Puntaje", "Sector", 
                        "P/E Trailing", "P/E Trailing (Prom. Sector)",
                        "ROE", "ROE (Prom. Sector)", "P/B", "P/B (Prom. Sector)",
                        "Market Cap", f"Perf ({periodo})", "Volatilidad (Anual)"
                    ]
                    st.dataframe(df_display[[c for c in cols_to_show if c in df_display.columns]], use_container_width=True)

                    # --- Gráfico de Ranking (similar al que ya tenías) ---
                    if len(df_display) > 1:
                        top_n = st.slider(
                            "Mostrar Top N en gráfico", 1, min(20, len(df_display)), min(10, len(df_display))
                        )
                        plot_df = df_display.head(top_n)
                        fig_rank = px.bar(
                            plot_df, x="Ticker", y="Puntaje", color="Sector",
                            title=f"Puntaje por Ticker (Top {top_n})",
                            hover_data=["Nombre", "Resultado", "P/E Trailing", "ROE", "P/B"]
                        )
                        st.plotly_chart(fig_rank, use_container_width=True)
                else:
                    progress.empty()
                    st.error("No se pudo obtener información válida para ninguno de los tickers ingresados.")
# Footer
st.caption(
    "Aviso: los datos provienen de Yahoo Finance vía yfinance. Algunos ratios pueden no estar disponibles o variar según el tipo de activo (acción vs ETF). "
    "Este contenido es solo con fines informativos y no constituye recomendación de inversión."
)
