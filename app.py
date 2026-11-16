from io import BytesIO
from google import genai
from openpyxl import load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import streamlit as st
from google.genai import types
import pandas as pd
import plotly.express as px
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=120000)
)

MODEL_NAME = "gemini-3.8-flash"

# Page settings
st.set_page_config(
    page_title="RetailIQ",
    page_icon="📊",
    layout="wide"
)

# Custom title
st.title("📊 RetailIQ")
st.caption("AI-Powered Retail Business Intelligence Platform")
st.divider()

# Upload dataset
st.sidebar.header("Upload Retail Data")
st.sidebar.divider()

st.sidebar.subheader("🤖 AI Business Copilot")

ai_access = st.sidebar.toggle(
    "Enable AI Assistant",
    value=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

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

        # Sidebar filters
        st.sidebar.subheader("Filters")

        regions = sorted(df["region"].dropna().unique())
        selected_regions = st.sidebar.multiselect(
            "Select Region",
            regions,
            default=regions
        )

        categories = sorted(df["category"].dropna().unique())
        selected_categories = st.sidebar.multiselect(
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

        left.plotly_chart(fig1, width="stretch")

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

        right.plotly_chart(fig2, width="stretch")

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

        st.plotly_chart(fig3, width="stretch")

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

        st.plotly_chart(fig4, width="stretch")

        # Customer insights
        st.subheader("Customer Insights")

        if "customer_segment" in filtered.columns:
            customer_segments = (
                filtered.groupby("customer_segment", as_index=False)
                ["sales_amount"].sum()
                .sort_values("sales_amount", ascending=False)
            )

            st.plotly_chart(
                px.bar(
                    customer_segments,
                    x="customer_segment",
                    y="sales_amount",
                    title="Revenue by Customer Segment"
                ),
                width="stretch"
            )
        else:
            st.info("Add a customer_segment column to the CSV to view segment-level customer insights.")

        # Inventory status (uses actual stock data only when provided)
        st.subheader("Inventory Status")
        stock_column = next(
            (col for col in ["stock_quantity", "inventory_quantity", "stock"] if col in filtered.columns),
            None
        )
        if stock_column:
            stock_data = filtered.copy()
            stock_data[stock_column] = pd.to_numeric(stock_data[stock_column], errors="coerce")
            inventory = (
                stock_data.groupby(["product_name", "category"], as_index=False)
                .agg(Available_Stock=(stock_column, "min"), Units_Sold=("quantity", "sum"))
                .sort_values("Available_Stock")
            )
            low_stock_limit = st.sidebar.number_input(
                "Low stock alert threshold", min_value=1, max_value=1000, value=10
            )
            low_stock = inventory[inventory["Available_Stock"] <= low_stock_limit]
            i1, i2 = st.columns(2)
            i1.metric("Products tracked", f"{len(inventory):,}")
            i2.metric("Low stock products", f"{len(low_stock):,}")
            if low_stock.empty:
                st.success("No products are below the selected stock threshold.")
            else:
                st.warning(f"{len(low_stock)} product(s) are at or below {low_stock_limit} units.")
                st.dataframe(low_stock, width="stretch", hide_index=True)
        else:
            st.info(
                "Inventory alerts need a stock_quantity (or inventory_quantity / stock) column in the uploaded CSV. "
                "Sales quantity alone cannot confirm current stock."
            )

        # Simple, explainable recommendations and demand estimate
        st.subheader("Business Recommendations & Demand Estimate")
        rec_col, forecast_col = st.columns(2)
        with rec_col:
            st.markdown("**Quick recommendations**")
            best_product = top_products.iloc[0]["product_name"] if not top_products.empty else None
            if best_product:
                st.write(f"• Review replenishment for **{best_product}**, currently the top product by units sold.")
            if len(category_sales) >= 2:
                weakest_category = category_sales.iloc[-1]["category"]
                st.write(f"• Review pricing, promotion, and assortment for **{weakest_category}**, the lowest-revenue category in the selected data.")
            if len(monthly_sales) >= 2:
                change = monthly_sales["sales_amount"].pct_change().iloc[-1] * 100
                direction = "increased" if change >= 0 else "decreased"
                st.write(f"• Revenue {direction} by **{abs(change):.1f}%** in the latest month versus the previous month; investigate the drivers.")
            st.caption("These are rule-based observations from the selected data, not generated AI recommendations.")
        with forecast_col:
            st.markdown("**Next-month demand estimate**")
            if len(monthly_sales) >= 2:
                demand_history = filtered.groupby("month", as_index=False)["quantity"].sum().sort_values("month")
                recent = demand_history["quantity"].tail(3)
                estimated_units = recent.mean()
                next_month = (pd.Period(monthly_sales["month"].max(), freq="M") + 1).strftime("%Y-%m")
                st.metric(f"Estimated units for {next_month}", f"{estimated_units:,.0f}")
                st.caption("Simple estimate: average units sold over the latest three available months. This is a baseline, not a trained forecasting model.")
            else:
                st.info("At least two months of sales data are needed to show a basic demand estimate.")

        # =====================================
        # RETAILIQ AI BUSINESS COPILOT
        # =====================================

        st.divider()
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

                st.session_state.retailiq_chat_history.append({
                    "role": "user",
                    "content": user_question
                })

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

                Important limitations:
                - Only uploaded and currently filtered dataset is available.
                - City allocation and profit/cost figures are simulated.
                - Do not claim simulated figures are actual business facts.
                - Do not invent missing business information.
                """

                # Prepare conversation history
                previous_chat = ""

                for msg in st.session_state.retailiq_chat_history[-11:-1]:
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
Clearly identify simulated data.

Recent conversation:
{previous_chat}

Business data:
{business_context}

User question:
{user_question}

Give a concise, accurate and useful answer.
"""


                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("RetailIQ AI is thinking..."):

                        try:
                           
                            response = client.models.generate_content(
                                model=MODEL_NAME,
                                contents=ai_prompt,
                                config=types.GenerateContentConfig(
                                    thinking_config=types.ThinkingConfig(
                                        thinking_level="LOW"
                                    ),
                                    max_output_tokens=512
                                )
)


                            answer = response.text

                            if not answer:
                                answer = "Sorry, I could not generate a response."

                            st.markdown(answer)

                            st.session_state.retailiq_chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                        except Exception as e:
                            st.error(
                                f"AI Error: {str(e)}"
                            )

        else:
            st.info("Enable AI Assistant from the sidebar to start chatting.")

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
            "City allocation, product costs and expenses below are "
            "simulated assumptions for project demonstration only."
        )

        # Demo city mapping by region
        city_mapping = {
            "North": ["Delhi", "Jaipur", "Chandigarh"],
            "South": ["Bengaluru", "Chennai", "Hyderabad"],
            "East": ["Kolkata", "Bhubaneswar", "Guwahati"],
            "West": ["Mumbai", "Pune", "Ahmedabad"],
            "Central": ["Bhopal", "Indore", "Raipur"]
        }
        # Fast vectorized Demo City allocation
        filtered["Demo City"] = "Other Demo City"

       
        for region, cities in city_mapping.items():
            mask = filtered["region"].eq(region)
            row_indexes = filtered.index[mask]

            filtered.loc[mask, "Demo City"] = [
                cities[i % len(cities)] for i in row_indexes
            ]


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
        # City-wise Sales Report
        city_report = (
            filtered.groupby("Demo City", as_index=False)
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
        if st.button("📊 Generate Excel Report"):
            
            # Excel report creation
            from io import BytesIO

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

                customer_report.to_excel(
                    writer, sheet_name="Customer Report", index=False
                )

                filtered.to_excel(
                    writer, sheet_name="Filtered Data", index=False
                )
                city_report.to_excel(
                    writer,
                    sheet_name="City Wise Sales",
                    index=False
                )

                profit_loss_report.to_excel(
                    writer,
                    sheet_name="Profit Loss",
                    index=False
                )
                # City report only if city column exists
                if "city" in filtered.columns:
                    city_report = (
                        filtered.groupby("city", as_index=False)
                        .agg(
                            Revenue=("sales_amount", "sum"),
                            Transactions=("transaction_id", "nunique"),
                            Units_Sold=("quantity", "sum")
                        )
                        .sort_values("Revenue", ascending=False)
                    )

                    city_report.to_excel(
                        writer, sheet_name="City Report", index=False
                    )

            # Add editable charts to Excel sheets
            from openpyxl.chart import BarChart, LineChart, Reference

            from openpyxl import load_workbook

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
            ws = wb["City Wise Sales"]
            chart = BarChart()
            chart.title = "Demo City-wise Revenue"
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
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter

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

            st.write("Download your complete business report:")

            st.download_button(
                label="📊 Download Complete Excel Report",
                data=excel_data,
                file_name="RetailIQ_Business_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="retailiq_filtered_data.csv",
                mime="text/csv"
            )

            st.success("RetailIQ dashboard and business reports loaded successfully!")
    except Exception as e:
        st.error(f"Unable to process dataset: {e}")

