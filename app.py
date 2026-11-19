import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from openpyxl import load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def linear_trend(y, steps=3):
    """Least-squares Linear Regression on time index. Returns predictions and R-squared."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = ((y - fitted) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    future = slope * np.arange(len(y), len(y) + steps) + intercept
    return future, r2



# Charts: no zoom / pan (stops the page zooming on touch). Only the camera (download photo)
# button stays; Streamlit's own fullscreen icon is left visible too.
CHART_CONFIG = {
    "displayModeBar": True,            # keep only the camera (download as photo) button
    "modeBarButtons": [["toImage"]],
    "toImageButtonOptions": {"format": "png", "filename": "retailiq_chart", "scale": 2},
    "scrollZoom": False,
    "doubleClick": False,
    "showTips": False,
    "displaylogo": False,
}


def plot_chart(target, fig, **kwargs):
    """Shows a Plotly chart that cannot be zoomed or dragged; clicking bars still works."""
    fig.update_layout(dragmode=False, hovermode="closest")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    kwargs.setdefault("config", CHART_CONFIG)
    return target.plotly_chart(fig, **kwargs)

def clicked_labels(event, labels):
    """Labels of the bars / pie slices the user clicked on a Plotly chart (on_select event)."""
    try:
        points = event.selection.points
    except Exception:
        return []
    labels = list(labels)
    found = []
    for pt in points:
        idx = pt.get("point_index", pt.get("point_number"))
        if isinstance(idx, int) and 0 <= idx < len(labels):
            found.append(labels[idx])
        else:
            val = pt.get("x", pt.get("label"))
            if val in labels:
                found.append(val)
    return list(dict.fromkeys(found))


def show_customer_table(data, title, key):
    """Customer list (ID, optional name, orders, units, revenue) for the clicked bar / slice."""
    st.markdown(f"##### 👥 {title}")
    if data.empty:
        st.info("No customers found for this selection.")
        return
    group_cols = ["customer_id"] + (["customer_name"] if "customer_name" in data.columns else [])
    table = (
        data.groupby(group_cols, as_index=False)
        .agg(Orders=("transaction_id", "nunique"), Units=("quantity", "sum"),
             Revenue=("sales_amount", "sum"), Last_Purchase=("transaction_date", "max"))
        .sort_values("Revenue", ascending=False)
    )
    table["Revenue"] = table["Revenue"].round(2)
    table["Last_Purchase"] = table["Last_Purchase"].dt.strftime("%Y-%m-%d")
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers", f"{len(table):,}")
    m2.metric("Orders", f"{int(table['Orders'].sum()):,}")
    m3.metric("Revenue", f"{table['Revenue'].sum():,.0f}")
    st.dataframe(table.head(100), width="stretch", hide_index=True)
    st.caption(f"Showing top {min(len(table), 100)} of {len(table):,} customers by revenue.")
    st.download_button(
        f"⬇️ Download all {len(table):,} customers (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}",
    )


def scroll_to_element(element_id):
    """Scrolls the page to the element with this id (used so the AI answer comes into view)."""
    js = f"""
    <script>
    // run {time.time_ns()}
    function go() {{
        const el = window.parent.document.getElementById("{element_id}");
        if (el) el.scrollIntoView({{behavior: "smooth", block: "start"}});
    }}
    [0, 250, 700, 1500].forEach(t => setTimeout(go, t));
    </script>
    """
    try:
        components.html(js, height=0)
    except Exception:
        try:
            st.html(js, unsafe_allow_javascript=True)
        except Exception:
            pass



def read_any_table(uploaded):
    """Reads CSV / TXT / Excel (xlsx, xls) / JSON uploads. Phones often give odd file types,
    so the content is tried in several ways instead of trusting only the extension."""
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()

    def as_csv():
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(BytesIO(raw), encoding=enc, sep=None, engine="python")
            except Exception:
                continue
        raise ValueError("not a readable CSV")

    def as_excel():
        sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
        # use the sheet that has the sales columns (e.g. 'Filtered Data' in a RetailIQ report)
        for sheet_df in sheets.values():
            if {"transaction_id", "sales_amount"}.issubset(sheet_df.columns):
                return sheet_df
        return next(iter(sheets.values()))

    def as_json():
        return pd.read_json(BytesIO(raw))

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        order = [as_excel, as_csv]
    elif name.endswith(".json"):
        order = [as_json, as_csv]
    else:
        order = [as_csv, as_excel, as_json]
    last = None
    for reader in order:
        try:
            df_read = reader()
            if df_read is not None and len(df_read.columns) > 1:
                return df_read
        except Exception as e:
            last = e
    raise ValueError(f"Unsupported or unreadable file ({last})")

# Gemini client (key is read from .streamlit/secrets.toml)
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=25000)  # 25 sec per try
)

# Change the model name in secrets.toml (GEMINI_MODEL) if needed
MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash")
# If the main model is slow/overloaded, these are tried next
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]


@st.cache_resource(show_spinner=False)
def discover_models():
    """Flash models available for this API key (looked up only once, only when needed)."""
    found = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "flash" in name and not any(
                x in name for x in ("image", "tts", "live", "audio", "embedding")
            ):
                found.append(name)
    except Exception:
        pass
    return found


def _stream_one(model, prompt):
    cfg = {"max_output_tokens": 2500}  # thinking tokens also count, so keep this high
    if "gemini-3" in model:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
    return client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**cfg),
    )


def stream_gemini(prompt):
    """Yield the answer piece by piece (text appears quickly). Falls back to other models."""
    models = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    last_error = None
    for round_no in range(2):
        for model in models:
            try:
                got_text = False
                hit_limit = False
                for chunk in _stream_one(model, prompt):
                    if chunk.text:
                        got_text = True
                        yield chunk.text
                    try:
                        if "MAX_TOKENS" in str(chunk.candidates[0].finish_reason):
                            hit_limit = True
                    except Exception:
                        pass
                if got_text and hit_limit:
                    yield "\n\n_(Answer was long and got cut. Ask: 'continue' or ask a narrower question.)_"
                if got_text:
                    return
            except Exception as e:
                last_error = e
                if got_text:       # already showed part of the answer, stop here
                    return
        if round_no == 0:          # first round failed: also try models found for this key
            models = [m for m in discover_models() if m not in models][:3]
            if not models:
                break
    # Last option: OpenAI API (only if OPENAI_API_KEY is added in secrets)
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        openai_key = ""
    if openai_key:
        yield from stream_openai(prompt, openai_key)
        return
    raise last_error or RuntimeError("No response from AI")


def stream_openai(prompt, api_key):
    """Optional second AI provider (OpenAI API)."""
    from openai import OpenAI
    oc = OpenAI(api_key=api_key, timeout=25)
    stream = oc.chat.completions.create(
        model=st.secrets.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        stream=True,
    )
    for ev in stream:
        if ev.choices and ev.choices[0].delta.content:
            yield ev.choices[0].delta.content


# ---------------- DATABASE: built-in SQLite by default, MySQL if MYSQL_URL is set in secrets ----------------
@st.cache_resource(show_spinner=False)
def get_engine():
    """Returns (engine, label). engine is None for SQLite (built-in, no extra library needed).
    MySQL needs: pip install sqlalchemy pymysql, and MYSQL_URL like mysql+pymysql://user:password@host:3306/dbname"""
    try:
        url = st.secrets.get("MYSQL_URL", "")
    except Exception:
        url = ""
    if url:
        try:
            from sqlalchemy import create_engine, text
            eng = create_engine(url, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(text("SELECT 1"))
            return eng, "MySQL"
        except Exception:
            pass  # fall back to SQLite
    return None, "SQLite"


@contextmanager
def db_conn():
    """Yields a connection that pandas can use (sqlite3 connection or SQLAlchemy connection)."""
    eng, _ = get_engine()
    if eng is None:
        conn = sqlite3.connect("retailiq.db")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    else:
        with eng.begin() as conn:
            yield conn


def db_execute(sql, params=None):
    eng, _ = get_engine()
    with db_conn() as conn:
        if eng is None:
            conn.execute(sql, params or {})
        else:
            from sqlalchemy import text
            conn.execute(text(sql), params or {})


def db_save_sales(df):
    with db_conn() as conn:
        df.to_sql("sales", conn, if_exists="replace", index=False, chunksize=2000)


def db_load_sales():
    try:
        with db_conn() as conn:
            return pd.read_sql("SELECT * FROM sales", conn)
    except Exception:
        return None  # table does not exist yet


def log_event(event):
    """Permanent usage log (stored in the database, survives page refresh)."""
    try:
        db_execute("CREATE TABLE IF NOT EXISTS usage_log (ts VARCHAR(30), event VARCHAR(50))")
        db_execute("INSERT INTO usage_log (ts, event) VALUES (:t, :e)",
                   {"t": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "e": event})
    except Exception:
        pass


def read_usage_log():
    try:
        with db_conn() as conn:
            return pd.read_sql("SELECT ts, event FROM usage_log ORDER BY ts DESC", conn)
    except Exception:
        return pd.DataFrame(columns=["ts", "event"])


# Page settings
st.set_page_config(
    page_title="RetailIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Keep touch scrolling smooth over charts
st.markdown(
    """
    <style>
    /* Finger on a chart = scroll the PAGE only. No pinch zoom, no double-tap zoom, no dragging the graph */
    .js-plotly-plot, .js-plotly-plot *,
    [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] * {
        touch-action: pan-y !important;
        -webkit-user-select: none; user-select: none;
        -webkit-tap-highlight-color: transparent;
    }
    html, body { overscroll-behavior-x: none; }
    /* Chart toolbar: show ONLY the download-image (camera) button */
    .modebar-group .modebar-btn { display: none !important; }
    .modebar-group .modebar-btn[data-title*="png" i],
    .modebar-group .modebar-btn[data-title*="image" i] { display: inline-block !important; }
    /* Streamlit toolbar: keep fullscreen, hide search / download icons (tables) */
    [data-testid="stElementToolbarButton"][aria-label*="earch" i],
    [data-testid="stElementToolbarButton"][aria-label*="ownload" i] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom title
st.title("📊 RetailIQ")
st.caption("AI-Powered Retail Business Intelligence Platform")
st.divider()

# ---------------- ADMIN MODULE: login + usage monitoring ----------------
if "auth" not in st.session_state:
    st.session_state.auth = False


# Data source section is on the MAIN page so it works on mobile too
st.subheader("📁 Data Source")
_, db_label = get_engine()
source = st.radio(
    "Choose data source",
    ["Upload CSV", "Use demo data (no upload needed)", f"Saved database ({db_label})"],
    horizontal=True
)
use_db = source.startswith("Saved")
use_demo = source.startswith("Use demo")

raw_df = None
data_name = ""
uploaded_file = None

if use_db:
    try:
        raw_df = db_load_sales()
    except Exception as e:
        st.error(f"Could not read the database: {e}")
    if raw_df is None:
        st.info("No data saved yet. Choose 'Upload CSV' once; every uploaded file is saved to the database automatically.")
    else:
        data_name = "database"
        st.caption(f"Loaded {len(raw_df):,} rows from the {db_label} database.")
        if st.button("🔄 Reload data from database"):
            st.rerun()
elif use_demo:
    demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retailiq_demo_with_inventory.csv")
    if os.path.exists(demo_path):
        raw_df = pd.read_csv(demo_path)
        data_name = "demo data"
        st.caption(f"Using built-in demo data ({len(raw_df):,} rows) with city, customer name and stock columns.")
    else:
        st.error("Demo file 'retailiq_demo_with_inventory.csv' was not found next to app.py. Add it to the GitHub repo.")
else:
    # No 'type' filter on purpose: phone file pickers grey out files when a type list is set.
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel - any file type works)"
    )
    if uploaded_file is None:
        st.info("Upload a CSV file above to see the dashboard. A sample file is available in the project repo (sample_retail_data.csv).")
    else:
        try:
            raw_df = read_any_table(uploaded_file)
            data_name = uploaded_file.name
        except Exception as e:
            st.error(f"Could not read this file: {e}")

ai_access = st.toggle(
    "🤖 Enable AI Business Copilot",
    value=True
)

if raw_df is not None:

    try:
        df = raw_df.copy()

        required_columns = [
            "transaction_id",
            "transaction_date",
            "customer_id",
            "product_name",
            "category",
            "quantity",
            "sales_amount",
            "region"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Clean data
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"], errors="coerce"
        )

        for col in ["quantity", "sales_amount", "unit_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

        df = df.dropna(
            subset=["transaction_date", "sales_amount"]
        )

        # DATABASE: every uploaded file is saved to the database (SQLite, or MySQL if configured)
        if not use_db and st.session_state.get("last_file") != data_name:
            st.session_state.last_file = data_name
            log_event("file_upload")
            try:
                db_save_sales(df)
            except Exception as e:
                st.warning(f"Could not save data to the database: {e}")

        # Sidebar filters
        with st.expander("🔎 Filters (Region / Category)", expanded=False):
            regions = sorted(df["region"].dropna().unique())
            selected_regions = st.multiselect(
                "Select Region",
                regions,
                default=regions
            )

            categories = sorted(df["category"].dropna().unique())
            selected_categories = st.multiselect(
                "Select Category",
                categories,
                default=categories
            )

        filtered = df[
            df["region"].isin(selected_regions)
            & df["category"].isin(selected_categories)
        ].copy()

        if filtered.empty:
            st.warning("No data available for selected filters.")
            st.stop()

        # Dashboard KPIs
        revenue = filtered["sales_amount"].sum()
        transactions = filtered["transaction_id"].nunique()
        customers = filtered["customer_id"].nunique()
        units = filtered["quantity"].sum()

        st.subheader("Business Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Revenue", f"{revenue:,.2f}")
        c2.metric("Transactions", f"{transactions:,}")
        c3.metric("Customers", f"{customers:,}")
        c4.metric("Units Sold", f"{units:,.0f}")

        st.divider()

        # Sales charts
        st.subheader("Sales Analytics")

        left, right = st.columns(2)

        category_sales = (
            filtered.groupby("category", as_index=False)
            ["sales_amount"].sum()
            .sort_values("sales_amount", ascending=False)
        )

        fig1 = px.bar(
            category_sales,
            x="category",
            y="sales_amount",
            title="Revenue by Category",
            color="category"
        )

        plot_chart(left, fig1, width="stretch")

        region_sales = (
            filtered.groupby("region", as_index=False)
            ["sales_amount"].sum()
        )

        fig2 = px.pie(
            region_sales,
            names="region",
            values="sales_amount",
            title="Region-wise Revenue",
            hole=0.4
        )

        plot_chart(right, fig2, width="stretch")

        # Monthly trend
        st.subheader("Monthly Sales Trend")

        filtered["month"] = (
            filtered["transaction_date"].dt.to_period("M").astype(str)
        )

        monthly_sales = (
            filtered.groupby("month", as_index=False)
            ["sales_amount"].sum()
        )

        fig3 = px.line(
            monthly_sales,
            x="month",
            y="sales_amount",
            markers=True,
            title="Monthly Revenue Trend"
        )

        plot_chart(st, fig3, width="stretch")

        # City-wise performance (real city column from the CSV)
        st.subheader("City-wise Performance")
        if "city" in filtered.columns:
            city_sales = (
                filtered.groupby("city", as_index=False)["sales_amount"]
                .sum().sort_values("sales_amount", ascending=False)
            )
            plot_chart(st, 
                px.bar(city_sales.head(10), x="city", y="sales_amount",
                       title="Top 10 Cities by Revenue", color="city"),
                width="stretch"
            )
        else:
            st.info("Add a 'city' column to the CSV to view city-wise performance.")

        # Top products
        st.subheader("Top Selling Products")

        top_products = (
            filtered.groupby("product_name", as_index=False)
            ["quantity"].sum()
            .sort_values("quantity", ascending=False)
            .head(10)
        )

        fig4 = px.bar(
            top_products,
            x="quantity",
            y="product_name",
            orientation="h",
            title="Top 10 Products by Units Sold"
        )

        plot_chart(st, fig4, width="stretch")

        # Top vs low performing products
        st.subheader("Top vs Low Performing Products")
        product_perf = (
            filtered.groupby("product_name", as_index=False)
            .agg(Revenue=("sales_amount", "sum"), Units_Sold=("quantity", "sum"))
            .sort_values("Revenue", ascending=False)
        )
        product_perf["Performance"] = "Average"
        if len(product_perf) >= 4:
            q_hi = product_perf["Revenue"].quantile(0.75)
            q_lo = product_perf["Revenue"].quantile(0.25)
            product_perf.loc[product_perf["Revenue"] >= q_hi, "Performance"] = "Top performer"
            product_perf.loc[product_perf["Revenue"] <= q_lo, "Performance"] = "Low performer"
            top10 = product_perf.head(10).sort_values("Revenue")
            low10 = product_perf.tail(10).sort_values("Revenue", ascending=False)
            pt, pl = st.columns(2)
            with pt:
                st.markdown("#### 🏆 Top 10 Best-Performing Products")
                st.caption("These products earn the HIGHEST revenue.")
                plot_chart(st, 
                    px.bar(top10, x="Revenue", y="product_name", orientation="h",
                           hover_data=["Units_Sold"], color_discrete_sequence=["#2ECC71"],
                           labels={"product_name": "Product"}),
                    width="stretch"
                )
            with pl:
                st.markdown("#### 📉 Top 10 Low-Performing Products")
                st.caption("These products earn the LOWEST revenue.")
                plot_chart(st, 
                    px.bar(low10, x="Revenue", y="product_name", orientation="h",
                           hover_data=["Units_Sold"], color_discrete_sequence=["#E74C3C"],
                           labels={"product_name": "Product"}),
                    width="stretch"
                )
            n_low = int((product_perf["Performance"] == "Low performer").sum())
            st.caption(f"Top performers = top 25% by revenue, low performers = bottom 25% ({n_low} products).")
            with st.expander("Full product performance table"):
                st.dataframe(product_perf, width="stretch", hide_index=True)
        else:
            st.info("At least 4 products are needed to compare top and low performers.")

        # Customer insights
        st.subheader("Customer Insights")

        if "customer_segment" in filtered.columns:
            customer_segments = (
                filtered.groupby("customer_segment", as_index=False)
                ["sales_amount"].sum()
                .sort_values("sales_amount", ascending=False)
            )
            seg_labels = customer_segments["customer_segment"].tolist()
            st.caption("👆 Click any bar to see the customers (ID, name, orders, revenue) in that segment.")
            seg_event = plot_chart(st, 
                px.bar(customer_segments, x="customer_segment", y="sales_amount",
                       title="Revenue by Customer Segment",
                       labels={"customer_segment": "Customer segment", "sales_amount": "Revenue"}),
                width="stretch", on_select="rerun", selection_mode="points", key="segment_chart"
            )
            seg_pick = clicked_labels(seg_event, seg_labels)
            if not seg_pick:
                chosen = st.selectbox("Or choose a segment from the list",
                                      ["— Select —"] + seg_labels, key="segment_select")
                seg_pick = [] if chosen == "— Select —" else [chosen]
            if seg_pick:
                show_customer_table(
                    filtered[filtered["customer_segment"].isin(seg_pick)],
                    f"Customers in segment: {', '.join(map(str, seg_pick))}", "segment_customers"
                )
        else:
            st.info("Add a customer_segment column to the CSV to view segment-level customer insights.")

        # Customer buying patterns
        st.subheader("Customer Buying Patterns")
        st.caption("👆 Click a bar or a pie slice to see the customers behind it.")
        pc1, pc2 = st.columns(2)
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_names = filtered["transaction_date"].dt.day_name()
        weekday_sales = (
            filtered.assign(weekday=day_names)
            .groupby("weekday", as_index=False)["sales_amount"].sum()
        )
        weekday_sales["weekday"] = pd.Categorical(weekday_sales["weekday"], weekday_order, ordered=True)
        weekday_sales = weekday_sales.sort_values("weekday")
        weekday_labels = weekday_sales["weekday"].astype(str).tolist()
        with pc1:
            weekday_event = plot_chart(st, 
                px.bar(weekday_sales, x="weekday", y="sales_amount", title="Revenue by Day of Week",
                       labels={"weekday": "Day", "sales_amount": "Revenue"}),
                width="stretch", on_select="rerun", selection_mode="points", key="weekday_chart"
            )
        orders_per_customer = filtered.groupby("customer_id")["transaction_id"].nunique()
        repeat_ids = orders_per_customer[orders_per_customer > 1].index
        one_time_ids = orders_per_customer[orders_per_customer == 1].index
        pie_labels = ["Repeat customers", "One-time customers"]
        pie_fig = px.pie(names=pie_labels, values=[len(repeat_ids), len(one_time_ids)],
                         hole=0.4, title="Repeat vs One-time Customers")
        pie_fig.update_traces(sort=False)
        with pc2:
            pie_event = plot_chart(st, 
                pie_fig, width="stretch", on_select="rerun", selection_mode="points", key="repeat_chart"
            )

        day_pick = clicked_labels(weekday_event, weekday_labels)
        if day_pick:
            day_data = filtered[day_names.isin(day_pick)]
            show_customer_table(day_data, f"Customers who bought on: {', '.join(day_pick)}", "weekday_customers")
            day_products = (
                day_data.groupby("product_name", as_index=False)
                .agg(Units=("quantity", "sum"), Revenue=("sales_amount", "sum"))
                .sort_values("Revenue", ascending=False).head(10)
            )
            st.markdown(f"**Top 10 products sold on {', '.join(day_pick)}**")
            st.dataframe(day_products, width="stretch", hide_index=True)

        group_pick = clicked_labels(pie_event, pie_labels)
        if group_pick:
            ids = []
            if "Repeat customers" in group_pick:
                ids += list(repeat_ids)
            if "One-time customers" in group_pick:
                ids += list(one_time_ids)
            show_customer_table(
                filtered[filtered["customer_id"].isin(ids)],
                f"{' + '.join(group_pick)}", "repeat_vs_onetime_customers"
            )

        st.markdown("**Top 10 Customers by Revenue**")
        top_group = ["customer_id"] + (["customer_name"] if "customer_name" in filtered.columns else [])
        top_customers = (
            filtered.groupby(top_group, as_index=False)
            .agg(Revenue=("sales_amount", "sum"), Orders=("transaction_id", "nunique"), Units=("quantity", "sum"))
            .sort_values("Revenue", ascending=False).head(10)
        )
        st.dataframe(top_customers, width="stretch", hide_index=True)

        # Inventory status (uses actual stock data only when provided)
        st.subheader("Inventory Status")
        stock_column = next(
            (col for col in ["stock_quantity", "inventory_quantity", "stock"] if col in filtered.columns),
            None
        )
        if stock_column:
            stock_data = filtered.copy()
            stock_data[stock_column] = pd.to_numeric(stock_data[stock_column], errors="coerce")
            stock_data = stock_data.dropna(subset=[stock_column]).sort_values("transaction_date")
            has_city = "city" in stock_data.columns
            inv_keys = ["product_name", "category"] + (["city"] if has_city else [])
            # latest known stock for every product (and every city, when a city column exists)
            inventory = (
                stock_data.groupby(inv_keys, as_index=False)
                .agg(Available_Stock=(stock_column, "last"), Units_Sold=("quantity", "sum"))
                .sort_values("Available_Stock")
            )
            days_in_data = max((filtered["transaction_date"].max() - filtered["transaction_date"].min()).days + 1, 1)
            inventory["Avg_Daily_Sales"] = (inventory["Units_Sold"] / days_in_data).round(2)
            inventory["Days_Stock_Left"] = (
                inventory["Available_Stock"] / inventory["Avg_Daily_Sales"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).round(1)
            low_stock_limit = st.number_input(
                "Low stock alert threshold", min_value=1, max_value=1000, value=10
            )
            low_stock = inventory[inventory["Available_Stock"] <= low_stock_limit]
            item_word = "Product-city items" if has_city else "Products"
            i1, i2 = st.columns(2)
            i1.metric(f"{item_word} tracked", f"{len(inventory):,}")
            i2.metric("Low stock items", f"{len(low_stock):,}")
            if low_stock.empty:
                st.success("No products are below the selected stock threshold.")
            else:
                st.warning(f"{len(low_stock)} item(s) are at or below {low_stock_limit} units.")
                st.dataframe(low_stock, width="stretch", hide_index=True)

            # City-wise inventory
            st.markdown("**City-wise Inventory**")
            if has_city:
                city_stock = (
                    inventory.groupby("city", as_index=False)
                    .agg(Total_Stock=("Available_Stock", "sum"),
                         Low_Stock_Items=("Available_Stock", lambda x: int((x <= low_stock_limit).sum())))
                    .sort_values("Total_Stock", ascending=False)
                )
                ci1, ci2 = st.columns(2)
                plot_chart(ci1, 
                    px.bar(city_stock, x="city", y="Total_Stock", title="Total Stock by City",
                           labels={"city": "City", "Total_Stock": "Units in stock"}),
                    width="stretch"
                )
                plot_chart(ci2, 
                    px.bar(city_stock.sort_values("Low_Stock_Items", ascending=False),
                           x="city", y="Low_Stock_Items", title="Low-stock Items by City",
                           color_discrete_sequence=["#E74C3C"],
                           labels={"city": "City", "Low_Stock_Items": "Low-stock items"}),
                    width="stretch"
                )
                pick_city = st.selectbox(
                    "View stock of a city", ["All cities"] + sorted(inventory["city"].dropna().unique().tolist()),
                    key="inventory_city"
                )
                city_view = inventory if pick_city == "All cities" else inventory[inventory["city"] == pick_city]
                st.dataframe(city_view.sort_values("Available_Stock"), width="stretch", hide_index=True)
            else:
                st.info("Add a 'city' column to the CSV to see city-wise inventory.")

            # Restock suggestions based on average daily sales
            plan_days = st.number_input(
                "Restock planning window (days of demand to cover)", min_value=7, max_value=90, value=30
            )
            inventory["Suggested_Reorder_Qty"] = np.ceil(
                np.maximum(inventory["Avg_Daily_Sales"] * plan_days - inventory["Available_Stock"], 0)
            ).astype(int)
            restock = inventory[inventory["Suggested_Reorder_Qty"] > 0].sort_values("Days_Stock_Left")
            st.markdown(f"**Restock suggestions (to cover the next {plan_days} days)**")
            if restock.empty:
                st.success("Current stock is enough to cover the planning window.")
            else:
                st.dataframe(restock.head(15), width="stretch", hide_index=True)
                st.caption("Suggested reorder quantity = average daily sales x planning days - available stock.")
        else:
            st.info(
                "Inventory needs a stock_quantity (or inventory_quantity / stock) column in the CSV, "
                "and a city column for city-wise inventory. Upload retailiq_demo_with_inventory.csv, "
                "or run enrich_retail_csv.py on your own file to add them."
            )

        # Simple, explainable recommendations and demand estimate
        st.subheader("Business Recommendations & Demand Estimate")
        rec_col, forecast_col = st.columns(2)
        with rec_col:
            st.markdown("**Quick recommendations**")
            best_product = top_products.iloc[0]["product_name"] if not top_products.empty else None
            if best_product:
                st.write(f"• Review replenishment for **{best_product}**, currently the top product by units sold.")
            low_names = product_perf[product_perf["Performance"] == "Low performer"]["product_name"].tail(3).tolist()
            if low_names:
                st.write(f"• Consider discounts, bundling or delisting for low performers: **{', '.join(low_names)}**.")
            if len(category_sales) >= 2:
                weakest_category = category_sales.iloc[-1]["category"]
                st.write(f"• Review pricing, promotion, and assortment for **{weakest_category}**, the lowest-revenue category in the selected data.")
            if len(monthly_sales) >= 2:
                change = monthly_sales["sales_amount"].pct_change().iloc[-1] * 100
                direction = "increased" if change >= 0 else "decreased"
                st.write(f"• Revenue {direction} by **{abs(change):.1f}%** in the latest month versus the previous month; investigate the drivers.")
            st.caption("These are rule-based observations from the selected data, not generated AI recommendations.")
        with forecast_col:
            st.markdown("**Sales Forecast (next 3 months)**")
            if len(monthly_sales) >= 4:
                preds, r2_score = linear_trend(monthly_sales["sales_amount"].values, 3)
                preds = np.maximum(preds, 0)
                last_period = pd.Period(monthly_sales["month"].max(), freq="M")
                future_months = [(last_period + i).strftime("%Y-%m") for i in range(1, 4)]
                forecast_df = pd.concat([
                    monthly_sales.assign(type="Actual"),
                    pd.DataFrame({"month": future_months, "sales_amount": preds, "type": "Forecast"}),
                ])
                plot_chart(st, 
                    px.line(forecast_df, x="month", y="sales_amount", color="type",
                            markers=True, title="Actual vs Forecast Revenue"),
                    width="stretch"
                )
                st.caption(f"Linear Regression trend model (R² = {r2_score:.2f}). Indicative forecast, not a guarantee.")
            else:
                st.info("At least 4 months of sales data are needed for the forecast.")

        # =====================================
        # RETAILIQ AI BUSINESS COPILOT
        # =====================================

        st.divider()
        st.markdown('<div id="retailiq-copilot" style="scroll-margin-top:80px"></div>', unsafe_allow_html=True)
        st.header("🤖 RetailIQ AI Business Copilot")

        st.caption(
            "Your multilingual AI assistant for business and general questions."
        )

        if ai_access:

            # Initialize chat history
            if "retailiq_chat_history" not in st.session_state:
                st.session_state.retailiq_chat_history = []

            # Display chat history
            for message in st.session_state.retailiq_chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Clear conversation
            if st.button("🗑️ Clear Chat History"):
                st.session_state.retailiq_chat_history = []
                st.rerun()

            # User question
            user_question = st.chat_input(
                "Ask anything in any language..."
            )

            if user_question:

                log_event("ai_question")
                st.session_state.retailiq_chat_history.append({
                    "role": "user",
                    "content": user_question
                })

                # Take the person to the answer: scroll from wherever they typed down to the chat
                st.markdown('<div id="retailiq-latest" style="scroll-margin-top:90px"></div>', unsafe_allow_html=True)
                scroll_to_element("retailiq-latest")

                with st.chat_message("user"):
                    st.markdown(user_question)

                # Prepare business data context
                top_products_ai = (
                    filtered.groupby("product_name")["sales_amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict()
                )

                region_sales_ai = (
                    filtered.groupby("region")["sales_amount"]
                    .sum()
                    .to_dict()
                )

                category_sales_ai = (
                    filtered.groupby("category")["sales_amount"]
                    .sum()
                    .to_dict()
                )

                monthly_sales_ai = (
                    filtered.groupby("month")["sales_amount"]
                    .sum()
                    .tail(12)
                    .to_dict()
                )

                extra_context = ""
                if "city" in filtered.columns:
                    city_ai = filtered.groupby("city")["sales_amount"].sum().sort_values(ascending=False).head(10).to_dict()
                    extra_context += f"\nTop Cities by Revenue:\n{city_ai}\n"
                if "customer_segment" in filtered.columns:
                    seg_ai = filtered.groupby("customer_segment")["sales_amount"].sum().to_dict()
                    extra_context += f"\nRevenue by Customer Segment:\n{seg_ai}\n"
                if stock_column:
                    extra_context += f"\nLow stock products (stock <= {low_stock_limit}): {low_stock['product_name'].head(10).tolist()}\n"

                business_context = f"""
                RETAILIQ BUSINESS DATA:

                Total Revenue: {revenue}
                Total Transactions: {transactions}
                Total Customers: {customers}
                Total Units Sold: {units}

                Top Products by Revenue:
                {top_products_ai}

                Region-wise Revenue:
                {region_sales_ai}

                Category-wise Revenue:
                {category_sales_ai}

                Monthly Revenue:
                {monthly_sales_ai}
                {extra_context}

                Important limitations:
                - Only uploaded and currently filtered dataset is available.
                - All figures above are real figures from the uploaded file.
                - Profit and cost are NOT in the data; do not state profit figures.
                - Do not invent missing business information.
                """

                # Prepare conversation history
                previous_chat = ""

                for msg in st.session_state.retailiq_chat_history[-7:-1]:
                    previous_chat += (
                        f"\n{msg['role']}: {msg['content']}"
                    )

                # Multilingual AI instructions
                
                
                # Lightweight multilingual AI prompt
                ai_prompt = f"""
You are RetailIQ AI, a helpful multilingual assistant.

Reply in the same language as the user.
Support Hindi, Hinglish, English and other languages.

For general questions:
Answer clearly and simply. Help with education,
technology, science, career and everyday topics.

For business questions:
Use the provided RetailIQ data.
Identify top products, sales, customers and regions.
Give practical suggestions.
Never invent business figures.
Do not add notes or disclaimers about simulated data; the figures given are real.

Recent conversation:
{previous_chat}

Business data:
{business_context}

User question:
{user_question}

Give a short, accurate answer (maximum 5-6 lines) unless the user asks for detail.
"""


                # Generate AI response
                with st.chat_message("assistant"):
                    with st.container():

                        try:
                           
                            answer = st.write_stream(stream_gemini(ai_prompt))



                            if not answer:
                                answer = "Sorry, I could not generate a response."

                            st.session_state.retailiq_chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                        except Exception as e:
                            st.error(
                                f"AI Error: {str(e)}"
                            )

        else:
            st.info("Turn on 'Enable AI Business Copilot' (near the top of the page) to start chatting.")

        # Data preview
        with st.expander("View Uploaded Data"):
            st.dataframe(filtered, width="stretch")

                # Business Reports
        st.divider()
        st.subheader("📥 Download Business Reports")

        # Region-wise report
        region_report = (
            filtered.groupby("region", as_index=False)
            .agg(
                Revenue=("sales_amount", "sum"),
                Transactions=("transaction_id", "nunique"),
                Units_Sold=("quantity", "sum"),
                Customers=("customer_id", "nunique")
            )
            .sort_values("Revenue", ascending=False)
        )

        # Category-wise report
        category_report = (
            filtered.groupby("category", as_index=False)
            .agg(
                Revenue=("sales_amount", "sum"),
                Units_Sold=("quantity", "sum")
            )
            .sort_values("Revenue", ascending=False)
        )

        # Product-wise report
        product_report = (
            filtered.groupby("product_name", as_index=False)
            .agg(
                Revenue=("sales_amount", "sum"),
                Units_Sold=("quantity", "sum")
            )
            .sort_values("Revenue", ascending=False)
        )

        # Monthly revenue and growth
        monthly_report = (
            filtered.groupby("month", as_index=False)
            ["sales_amount"].sum()
            .rename(columns={"sales_amount": "Revenue"})
            .sort_values("month")
        )

        monthly_report["Growth_Percent"] = (
            monthly_report["Revenue"].pct_change() * 100
        ).fillna(0).round(2)

        # Customer-wise report
        customer_report = (
            filtered.groupby("customer_id", as_index=False)
            .agg(
                Revenue=("sales_amount", "sum"),
                Transactions=("transaction_id", "nunique"),
                Units_Sold=("quantity", "sum")
            )
            .sort_values("Revenue", ascending=False)
        )

        # SIMULATED DEMO DATA
        st.warning(
            "Product costs and expenses in the Profit/Loss sheet are "
            "illustrative assumptions (70% cost, 10% expenses) for demonstration only."
        )

        # Illustrative cost and expense assumptions
        filtered["Estimated Product Cost"] = (
            filtered["sales_amount"] * 0.70
        )

        filtered["Estimated Gross Profit"] = (
            filtered["sales_amount"]
            - filtered["Estimated Product Cost"]
        )

        filtered["Estimated Operating Expense"] = (
            filtered["sales_amount"] * 0.10
        )

        filtered["Estimated Net Profit"] = (
            filtered["Estimated Gross Profit"]
            - filtered["Estimated Operating Expense"]
        )


        # Business summary
        summary_report = pd.DataFrame({
            "Metric": [
                "Total Revenue",
                "Total Transactions",
                "Total Customers",
                "Total Units Sold"
            ],
            "Value": [
                revenue,
                transactions,
                customers,
                units
            ]
        })
        # City-wise Sales Report (only if a real city column exists)
        city_report = None
        if "city" in filtered.columns:
            city_report = (
                filtered.groupby("city", as_index=False)
                .agg(
                    Revenue=("sales_amount", "sum"),
                    Units_Sold=("quantity", "sum"),
                    Transactions=("transaction_id", "nunique")
                )
                .sort_values("Revenue", ascending=False)
            )

        # Profit and Loss Report
        profit_loss_report = pd.DataFrame({
            "Metric": [
                "Total Revenue",
                "Estimated Product Cost",
                "Estimated Gross Profit",
                "Estimated Operating Expenses",
                "Estimated Net Profit"
            ],
            "Amount": [
                filtered["sales_amount"].sum(),
                filtered["Estimated Product Cost"].sum(),
                filtered["Estimated Gross Profit"].sum(),
                filtered["Estimated Operating Expense"].sum(),
                filtered["Estimated Net Profit"].sum()
            ]
        })
        def build_excel_report():
            """Builds the Excel file only when the user clicks Download."""
            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                summary_report.to_excel(
                    writer, sheet_name="Business Summary", index=False
                )

                region_report.to_excel(
                    writer, sheet_name="Region Report", index=False
                )

                category_report.to_excel(
                    writer, sheet_name="Category Report", index=False
                )

                product_report.to_excel(
                    writer, sheet_name="Product Report", index=False
                )

                monthly_report.to_excel(
                    writer, sheet_name="Monthly Growth", index=False
                )

                # Large sheets are capped so the report is built quickly
                customer_report.head(1000).to_excel(
                    writer, sheet_name="Customer Report", index=False
                )

                filtered.head(5000).to_excel(
                    writer, sheet_name="Filtered Data", index=False
                )
                if city_report is not None:
                    city_report.to_excel(
                        writer, sheet_name="City Wise Sales", index=False
                    )

                profit_loss_report.to_excel(
                    writer,
                    sheet_name="Profit Loss",
                    index=False
                )

            # Add editable charts to Excel sheets
            excel_buffer.seek(0)
            wb = load_workbook(excel_buffer)

            # Region-wise Revenue Chart
            ws = wb["Region Report"]
            chart = BarChart()
            chart.title = "Region-wise Revenue"
            chart.y_axis.title = "Revenue"
            chart.x_axis.title = "Region"
            chart.add_data(
                Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                titles_from_data=True
            )
            chart.set_categories(
                Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            )
            ws.add_chart(chart, "G2")

            # Category-wise Sales Chart
            ws = wb["Category Report"]
            chart = BarChart()
            chart.title = "Category-wise Revenue"
            chart.add_data(
                Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                titles_from_data=True
            )
            chart.set_categories(
                Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            )
            ws.add_chart(chart, "G2")

            # Product-wise Sales Chart
            ws = wb["Product Report"]
            chart = BarChart()
            chart.title = "Product-wise Revenue"
            chart.add_data(
                Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                titles_from_data=True
            )
            chart.set_categories(
                Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            )
            ws.add_chart(chart, "G2")

            # City-wise Sales Chart
            if "City Wise Sales" in wb.sheetnames:
                ws = wb["City Wise Sales"]
                chart = BarChart()
                chart.title = "City-wise Revenue"
                chart.add_data(
                    Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                    titles_from_data=True
                )
                chart.set_categories(
                    Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                )
                ws.add_chart(chart, "G2")

            # Monthly Revenue Trend
            ws = wb["Monthly Growth"]
            chart = LineChart()
            chart.title = "Monthly Revenue Growth"
            chart.y_axis.title = "Revenue"
            chart.x_axis.title = "Month"
            chart.add_data(
                Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                titles_from_data=True
            )
            chart.set_categories(
                Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            )
            ws.add_chart(chart, "G2")

            # Profit and Loss Chart
            ws = wb["Profit Loss"]
            chart = BarChart()
            chart.title = "Estimated Profit and Loss"
            chart.y_axis.title = "Amount"
            chart.add_data(
                Reference(ws, min_col=2, min_row=1, max_row=ws.max_row),
                titles_from_data=True
            )
            chart.set_categories(
                Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            )
            ws.add_chart(chart, "D2")
        
            # Executive Dashboard Sheet
            ws_dash = wb.create_sheet("Executive Dashboard", 0)

            # Dashboard Title
            ws_dash.merge_cells("A1:H1")
            ws_dash["A1"] = "RetailIQ - Executive Business Dashboard"
            ws_dash["A1"].font = Font(
                bold=True, size=18, color="FFFFFF"
            )
            ws_dash["A1"].fill = PatternFill(
                fill_type="solid", fgColor="17365D"
            )
            ws_dash["A1"].alignment = Alignment(
                horizontal="center"
            )
            ws_dash.row_dimensions[1].height = 35

            # Business KPIs
            ws_dash.append([])
            ws_dash.append(["Business Metric", "Performance"])
            ws_dash.append(["Total Revenue", revenue])
            ws_dash.append(["Total Transactions", transactions])
            ws_dash.append(["Total Customers", customers])
            ws_dash.append(["Total Units Sold", units])

            # Category Performance
            ws_dash["A10"] = "Category"
            ws_dash["B10"] = "Revenue"

            for i, row in enumerate(
                category_report.head(10).itertuples(index=False),
                start=11
            ):
                ws_dash.cell(i, 1, row[0])
                ws_dash.cell(i, 2, row[1])

            # Top 5 Products
            ws_dash["D10"] = "Top Products"
            ws_dash["E10"] = "Revenue"

            for i, row in enumerate(
                product_report.head(5).itertuples(index=False),
                start=11
            ):
                ws_dash.cell(i, 4, row[0])
                ws_dash.cell(i, 5, row[1])

            # Region Revenue Chart
            region_chart = BarChart()
            region_chart.title = "Region-wise Revenue"
            region_chart.y_axis.title = "Revenue"

            region_ws = wb["Region Report"]
            region_chart.add_data(
                Reference(
                    region_ws,
                    min_col=2,
                    min_row=1,
                    max_row=region_ws.max_row
                ),
                titles_from_data=True
            )
            region_chart.set_categories(
                Reference(
                    region_ws,
                    min_col=1,
                    min_row=2,
                    max_row=region_ws.max_row
                )
            )
            ws_dash.add_chart(region_chart, "G3")

            # Monthly Revenue Trend
            monthly_chart = LineChart()
            monthly_chart.title = "Monthly Revenue Trend"

            monthly_ws = wb["Monthly Growth"]
            monthly_chart.add_data(
                Reference(
                    monthly_ws,
                    min_col=2,
                    min_row=1,
                    max_row=monthly_ws.max_row
                ),
                titles_from_data=True
            )
            monthly_chart.set_categories(
                Reference(
                    monthly_ws,
                    min_col=1,
                    min_row=2,
                    max_row=monthly_ws.max_row
                )
            )
            ws_dash.add_chart(monthly_chart, "G20")

            # Top Products Chart
            product_chart = BarChart()
            product_chart.title = "Top 5 Products by Revenue"

            product_chart.add_data(
                Reference(ws_dash, min_col=5, min_row=10, max_row=15),
                titles_from_data=True
            )
            product_chart.set_categories(
                Reference(ws_dash, min_col=4, min_row=11, max_row=15)
            )
            ws_dash.add_chart(product_chart, "D20")

            # Dashboard formatting
            for cell in ["A3", "B3", "A10", "B10", "D10", "E10"]:
                ws_dash[cell].font = Font(
                    bold=True, color="FFFFFF"
                )
                ws_dash[cell].fill = PatternFill(
                    fill_type="solid", fgColor="4472C4"
                )

            ws_dash["B4"].number_format = '₹#,##0.00'
            ws_dash.column_dimensions["A"].width = 25
            ws_dash.column_dimensions["B"].width = 20
            ws_dash.column_dimensions["D"].width = 30
            ws_dash.column_dimensions["E"].width = 20

            # Professional Excel Report Formatting
            header_fill = PatternFill(
                fill_type="solid",
                fgColor="17365D"
            )
            header_font = Font(
                bold=True,
                color="FFFFFF",
                size=11
            )
            thin_border = Border(
                bottom=Side(style="thin", color="D9E2F3")
            )

            
            for ws in wb.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(
                        horizontal="center",
                        vertical="center"
                    )

                ws.row_dimensions[1].height = 25

                if ws.title == "Filtered Data":
                    for col in ws.columns:
                        col_letter = get_column_letter(col[0].column)
                        ws.column_dimensions[col_letter].width = 18
                    continue

                for col in ws.columns:
                    col_letter = get_column_letter(col[0].column)
                    header = str(col[0].value or "").lower()

                    for cell in col[1:]:
                        cell.border = thin_border
                        cell.alignment = Alignment(vertical="center")

                        if isinstance(cell.value, (int, float)):
                            if "percent" in header or "pct" in header:
                                cell.number_format = '0.00"%"'
                            elif any(word in header for word in [
                                "revenue", "sales", "price",
                                "profit", "cost", "expense", "amount"
                            ]):
                                cell.number_format = '₹#,##0.00'

                    max_length = max(
                        (len(str(cell.value or "")) for cell in col[:100]),
                        default=10
                    )

                    ws.column_dimensions[col_letter].width = min(
                        max(max_length + 3, 12), 32
                    )


            final_excel_buffer = BytesIO()
            wb.save(final_excel_buffer)
            excel_data = final_excel_buffer.getvalue()
            return excel_data

        st.caption("The Excel report includes the top 1,000 customers and the first 5,000 data rows. Full data is in the CSV downloads below.")
        st.download_button(
            label="📊 Download Complete Excel Report",
            data=build_excel_report,
            file_name="RetailIQ_Business_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore",
        )


        st.divider()
        st.write("Download individual CSV reports:")

        d1, d2, d3 = st.columns(3)

        d1.download_button(
            "Region Report CSV",
            region_report.to_csv(index=False).encode("utf-8"),
            "region_report.csv",
            "text/csv"
        )

        d2.download_button(
            "Category Report CSV",
            category_report.to_csv(index=False).encode("utf-8"),
            "category_report.csv",
            "text/csv"
        )

        d3.download_button(
            "Monthly Growth CSV",
            monthly_report.to_csv(index=False).encode("utf-8"),
            "monthly_growth.csv",
            "text/csv"
        )

        st.download_button(
            "Download Complete Filtered CSV",
            data=lambda: filtered.to_csv(index=False).encode("utf-8"),
            file_name="retailiq_filtered_data.csv",
            mime="text/csv",
            on_click="ignore"
        )

        def build_powerbi_csv():
            pbi = filtered[[c for c in filtered.columns if not c.startswith("Estimated")]].copy()
            pbi["Year"] = pbi["transaction_date"].dt.year
            pbi["Month"] = pbi["transaction_date"].dt.month
            pbi["Month_Name"] = pbi["transaction_date"].dt.month_name()
            pbi["Weekday"] = pbi["transaction_date"].dt.day_name()
            return pbi.to_csv(index=False).encode("utf-8")

        st.markdown("**📈 Power BI**")
        st.download_button(
            "Download Power BI-ready dataset (CSV)",
            data=build_powerbi_csv,
            file_name="retailiq_powerbi_dataset.csv",
            mime="text/csv",
            on_click="ignore"
        )
        st.caption("In Power BI Desktop: Get Data > Text/CSV > select this file, then build your own dashboards.")

        st.success("RetailIQ dashboard and business reports loaded successfully!")

        # ADMIN PANEL (password protected; the rest of the app is open to everyone)
        with st.expander("🛠️ Admin Panel (password protected)"):
            if not st.session_state.auth:
                with st.form("admin_login"):
                    admin_pw = st.text_input("Admin password", type="password")
                    ok = st.form_submit_button("Unlock Admin Panel")
                if ok:
                    if admin_pw.strip() == str(st.secrets.get("ADMIN_PASSWORD", "admin123")).strip():
                        st.session_state.auth = True
                        st.rerun()
                    else:
                        st.error("Wrong password")
            else:
                usage_log = read_usage_log()
                a1, a2, a3 = st.columns(3)
                a1.metric("Files uploaded", int((usage_log["event"] == "file_upload").sum()))
                a2.metric("AI questions asked", int((usage_log["event"] == "ai_question").sum()))
                a3.metric("Total events", len(usage_log))
                _, db_name = get_engine()
                try:
                    with db_conn() as conn:
                        n_rows = int(pd.read_sql("SELECT COUNT(*) AS n FROM sales", conn)["n"].iloc[0])
                except Exception:
                    n_rows = 0
                st.write(f"Data source: **{db_name}** database, table `sales` ({n_rows:,} rows). Usage is stored permanently in table `usage_log`.")
                if not usage_log.empty:
                    st.markdown("**Recent activity**")
                    st.dataframe(usage_log.head(15), width="stretch", hide_index=True)
                if st.button("Lock Admin Panel"):
                    st.session_state.auth = False
                    st.rerun()
    except Exception as e:
        st.error(f"Unable to process dataset: {e}")

