# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
tab1, tab2 = st.tabs(["🔍 Análisis individual", "🏁 Comparador"])

with tab1:
    if ticker:
        df, decision, razones, meta, _t = analizar_ticker(ticker, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb)

        st.subheader(f"🪪 {meta['long_name']} ({ticker})")
        
        hist = get_history(ticker, periodo, intervalo)
        period_performance = None
        if not hist.empty and "Close" in hist.columns and len(hist) > 1:
            first_close = float(hist["Close"].iloc[0])
            last_close = float(hist["Close"].iloc[-1])
            if first_close > 0:
                period_performance = (last_close / first_close) - 1.0
        
        cols = st.columns(5)
        cols[0].metric("Resultado", decision)
        cols[1].metric("Puntaje", f"{meta['score']*100:.0f} %")
        cols[2].metric(f"Cambio {periodo}", 
                      f"{period_performance:.2%}" if period_performance is not None else "—",
                      delta=f"{period_performance:.2%}" if period_performance is not None else None)
        cols[3].metric("Moneda", meta["currency"])
        cols[4].metric("Exchange", meta["exchange"])
        
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

        st.subheader("📌 Evaluación")
        st.write("**Explicación de métricas:**")
        for r in razones:
            st.write(f"- {r}")

        st.subheader("📉 Evolución del precio")
        if hist.empty:
            st.error("No hay datos históricos disponibles para los parámetros seleccionados.")
        else:
            chart_title = f"Precio de cierre - {ticker} ({periodo}, {intervalo})"
            if period_performance is not None:
                chart_title += f" | Cambio: {period_performance:.2%}"
            
            plot_df = hist.copy()
            if sma50_on and len(plot_df) >= 50:
                plot_df["SMA50"] = plot_df["Close"].rolling(50).mean()
            if sma200_on and len(plot_df) >= 200:
                plot_df["SMA200"] = plot_df["Close"].rolling(200).mean()

            fig = px.line(
                plot_df,
                x=plot_df.index,
                y=["Close"] + [c for c in ["SMA50", "SMA200"] if c in plot_df.columns],
                title=chart_title
            )
            if log_scale:
                fig.update_yaxes(type="log")
            fig.update_layout(xaxis_title="Fecha", yaxis_title=f"Precio ({meta['currency']})", legend_title="Serie")
            st.plotly_chart(fig, use_container_width=True)

            if not hist.empty:
                min_price = hist["Close"].min()
                max_price = hist["Close"].max()
                current_price = hist["Close"].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"{current_price:.2f} {meta['currency']}")
                col2.metric("Mínimo del Período", f"{min_price:.2f} {meta['currency']}")
                col3.metric("Máximo del Período", f"{max_price:.2f} {meta['currency']}")

            if rsi_on:
                close = plot_df["Close"]
                delta = close.diff()
                up = delta.clip(lower=0.0)
                down = -1 * delta.clip(upper=0.0)
                roll = 14
                roll_up = up.rolling(roll).mean()
                roll_down = down.rolling(roll).mean()
                rs = roll_up / roll_down
                rsi = 100 - (100 / (1 + rs))
                rsi_df = pd.DataFrame({"RSI": rsi}).dropna()
                fig_rsi = px.line(rsi_df, x=rsi_df.index, y="RSI", title=f"RSI (14) - {ticker}")
                fig_rsi.update_layout(xaxis_title="Fecha", yaxis_title="RSI")
                fig_rsi.add_hrect(y0=30, y1=70, line_width=0, fillcolor="LightGray", opacity=0.15)
                st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    st.header("📊 Comparador de Tickers (ranking por puntaje)")
    st.write("Ingresá una lista separada por comas. Ej: `AAPL, MSFT, SPY, NVDA`")
    tickers_input = st.text_area("Tickers", value="AAPL, MSFT, SPY").strip()

    st.caption("El comparador utiliza los mismos umbrales, pesos y período/intervalo elegidos en la barra lateral.")

    if tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) > 0:
            progress = st.progress(0.0, text="Analizando…")
            rows = []

            for i, tk in enumerate(tickers, start=1):
                try:
                    df_i, decision_i, razones_i, meta_i, _t = analizar_ticker(
                        tk, pe_thr, roe_thr, pb_thr, w_pe, w_roe, w_pb
                    )

                    # Performance en el período
                    hist_i = get_history(tk, periodo, intervalo)
                    perf = None
                    if not hist_i.empty and "Close" in hist_i:
                        first_close = float(hist_i["Close"].iloc[0])
                        last_close = float(hist_i["Close"].iloc[-1])
                        if first_close > 0:
                            perf = (last_close / first_close) - 1.0

                    rows.append({
                        "Ticker": tk,
                        "Nombre": meta_i["long_name"],
                        "Resultado": decision_i,
                        "Puntaje": round(meta_i["score"] * 100, 1),
                        "P/E Trailing": df_i["P/E Trailing"].iloc[0],
                        "ROE": df_i["ROE"].iloc[0],
                        "P/B": df_i["P/B"].iloc[0],
                        "Market Cap": df_i["Market Cap"].iloc[0],
                        "Moneda": meta_i["currency"],
                        "Exchange": meta_i["exchange"],
                        "Sector": meta_i["sector"],
                        "Industria": meta_i["industry"],
                        f"Perf ({periodo})": f"{perf:.2%}" if perf is not None else "—",
                    })
                except Exception:
                    rows.append({
                        "Ticker": tk,
                        "Nombre": "—",
                        "Resultado": "Error",
                        "Puntaje": 0.0,
                        "P/E Trailing": "—",
                        "ROE": "—",
                        "P/B": "—",
                        "Market Cap": "—",
                        "Moneda": "—",
                        "Exchange": "—",
                        "Sector": "—",
                        "Industria": "—",
                        f"Perf ({periodo})": "—",
                    })
                finally:
                    progress.progress(i / len(tickers), text=f"Analizando {tk} ({i}/{len(tickers)})")

            progress.empty()

            if len(rows) == 0:
                st.warning("No se pudo analizar ningún ticker.")
            else:
                rank_df = pd.DataFrame(rows)
                rank_df_sorted = rank_df.sort_values(by="Puntaje", ascending=False, ignore_index=True)

                st.subheader("🏆 Ranking")
                st.dataframe(rank_df_sorted, use_container_width=True)

                # --- Slider robusto Top N ---
                max_val = min(20, len(rank_df_sorted))
                if max_val <= 1:
                    st.info("Solo hay un ticker en el ranking, no se genera gráfico Top N.")
                    top_n = 1
                else:
                    min_val = 1
                    default_val = min(10, max_val)
                    top_n = st.slider(
                        "Mostrar top N en gráfico",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )

                # Gráfico solo si hay 2+ y top_n >= 1
                if max_val > 1 and top_n >= 1:
                    plot_df = rank_df_sorted.head(top_n)
                    fig_rank = px.bar(
                        plot_df,
                        x="Ticker",
                        y="Puntaje",
                        hover_data=["Nombre", "Resultado", f"Perf ({periodo})", "P/E Trailing", "ROE", "P/B"],
                        title=f"Puntaje por ticker (Top {top_n})",
                    )
                    fig_rank.update_layout(yaxis_title="Puntaje (%)", xaxis_title="Ticker")
                    st.plotly_chart(fig_rank, use_container_width=True)

                # Filtros rápidos
                with st.expander("🔎 Filtros rápidos"):
                    colf1, colf2, colf3 = st.columns(3)
                    min_score = colf1.slider("Puntaje mínimo (%)", 0, 100, 0)
                    solo_acciones = colf2.checkbox("Excluir ETFs", value=False)
                    req_perf_pos = colf3.checkbox(f"Exigir Performance {periodo} > 0%", value=False)

                    filtered = rank_df_sorted.copy()
                    filtered = filtered[filtered["Puntaje"] >= min_score]

                    if solo_acciones:
                        mask_etf = filtered["Nombre"].str.contains("ETF", case=False, na=False) | (filtered["Sector"] == "—")
                        filtered = filtered[~mask_etf]

                    def perf_val(s):
                        try:
                            return float(s.replace("%", "")) / 100.0
                        except Exception:
                            return None

                    if req_perf_pos:
                        perf_values = filtered[f"Perf ({periodo})"].apply(perf_val)
                        filtered = filtered[perf_values > 0]

                    st.write("**Resultado filtrado:**")
                    st.dataframe(filtered, use_container_width=True)

# Footer
st.caption(
    "Aviso: los datos provienen de Yahoo Finance vía yfinance. Algunos ratios pueden no estar disponibles o variar según el tipo de activo (acción vs ETF). "
    "Este contenido es solo con fines informativos y no constituye recomendación de inversión."
)
