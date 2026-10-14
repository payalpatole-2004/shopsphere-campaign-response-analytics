import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sidebar import show_sidebar


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ShopSphere | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Common ShopSphere sidebar
show_sidebar()


# =========================================================
# PROJECT PATH
# =========================================================

# This file is inside the pages folder,
# so parents[1] points to the main project folder.

BASE = Path(__file__).parents[1]


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data_path = BASE / "data" / "campaign_customer_data.csv"

    if not data_path.exists():
        return None

    return pd.read_csv(data_path)


df = load_data()


# =========================================================
# CHECK DATA
# =========================================================

if df is None:

    st.error("❌ Data file not found.")

    st.info(
        "Please make sure campaign_customer_data.csv "
        "is inside the data folder."
    )

    st.stop()


# =========================================================
# TARGET
# =========================================================

target = "Campaign_Purchased"


if target not in df.columns:

    st.error(
        f"❌ Required column '{target}' was not found "
        "in the dataset."
    )

    st.stop()


# =========================================================
# PAGE HEADER
# =========================================================

st.title("📊 Campaign Analytics Dashboard")

st.caption(
    "Customer, campaign and purchase response analytics"
)


st.divider()


# =========================================================
# KPI CALCULATIONS
# =========================================================

# Conversion Rate
conversion_rate = df[target].mean() * 100


# Purchases
purchase_count = int(df[target].sum())


# Customers
if "Customer_ID" in df.columns:

    customer_count = df["Customer_ID"].nunique()

else:

    customer_count = len(df)


# Average Order Value
if "Avg_Order_Value" in df.columns:

    avg_order_value = df["Avg_Order_Value"].mean()

else:

    avg_order_value = None


# =========================================================
# CAMPAIGN PERFORMANCE
# =========================================================

st.header("📈 Campaign Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎯 Conversion Rate",
        f"{conversion_rate:.2f}%"
    )


with col2:

    st.metric(
        "🛒 Purchases",
        f"{purchase_count:,}"
    )


with col3:

    st.metric(
        "👥 Customers",
        f"{customer_count:,}"
    )


with col4:

    if avg_order_value is not None:

        st.metric(
            "💰 Avg Order Value",
            f"₹{avg_order_value:,.0f}"
        )

    else:

        st.metric(
            "💰 Avg Order Value",
            "N/A"
        )


st.divider()


# =========================================================
# ROW 1 — PURCHASE DISTRIBUTION + CUSTOMER SEGMENT
# =========================================================

left, right = st.columns(2)


# =========================================================
# 1. PURCHASE DISTRIBUTION
# =========================================================

with left:

    st.subheader("🛒 Purchase Distribution")


    counts = (
        df[target]
        .value_counts()
        .rename(
            index={
                0: "Not Purchased",
                1: "Purchased"
            }
        )
        .reset_index()
    )


    counts.columns = [
        "Purchase Status",
        "Customers"
    ]


    fig_purchase = px.pie(
        counts,
        values="Customers",
        names="Purchase Status",
        hole=0.45,
        color="Purchase Status",
        color_discrete_map={
            "Purchased": "#22C55E",
            "Not Purchased": "#CBD5E1"
        }
    )


    fig_purchase.update_traces(
        textinfo="percent+label",
        textfont_size=14
    )


    fig_purchase.update_layout(
        height=420,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),
        legend_title_text=""
    )


    st.plotly_chart(
        fig_purchase,
        use_container_width=True
    )


# =========================================================
# 2. CUSTOMER SEGMENT
# =========================================================

with right:

    st.subheader("👥 Purchase Rate by Customer Segment")


    if "Customer_Segment" in df.columns:

        segment_data = (
            df.groupby("Customer_Segment")[target]
            .mean()
            .mul(100)
            .reset_index()
        )


        segment_data.columns = [
            "Customer Segment",
            "Purchase Rate"
        ]


        fig_segment = px.bar(
            segment_data,
            x="Customer Segment",
            y="Purchase Rate",
            color="Customer Segment",
            text_auto=".2f",
            color_discrete_sequence=[
                "#6366F1",
                "#14B8A6",
                "#F59E0B",
                "#EC4899",
                "#8B5CF6"
            ]
        )


        fig_segment.update_traces(
            textposition="outside"
        )


        fig_segment.update_layout(
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            xaxis_title="Customer Segment",
            yaxis_title="Purchase Rate (%)",
            showlegend=False
        )


        st.plotly_chart(
            fig_segment,
            use_container_width=True
        )


    else:

        st.warning(
            "Customer_Segment column is not available."
        )


st.divider()


# =========================================================
# ROW 2 — CAMPAIGN TYPE + PREFERRED CHANNEL
# =========================================================

left, right = st.columns(2)


# =========================================================
# 3. PURCHASE RATE BY CAMPAIGN TYPE
# =========================================================

with left:

    st.subheader("📣 Purchase Rate by Campaign Type")


    if "Campaign_Type" in df.columns:

        summary = (
            df.groupby("Campaign_Type")
            .agg(
                Customers=(
                    "Customer_ID",
                    "nunique"
                )
                if "Customer_ID" in df.columns
                else (
                    target,
                    "count"
                ),

                Purchase_Rate=(
                    target,
                    "mean"
                )
            )
            .reset_index()
        )


        summary["Purchase_Rate"] = (
            summary["Purchase_Rate"] * 100
        )


        fig_campaign = px.bar(
            summary,
            x="Campaign_Type",
            y="Purchase_Rate",
            text_auto=".2f",
             color_discrete_sequence=[
                "#EC4899"
            ]
        )


        fig_campaign.update_traces(
            textposition="outside"
        )


        fig_campaign.update_layout(
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            xaxis_title="Campaign Type",
            yaxis_title="Purchase Rate (%)"
        )


        st.plotly_chart(
            fig_campaign,
            use_container_width=True
        )


    else:

        st.warning(
            "Campaign_Type column is not available."
        )


# =========================================================
# 4. PURCHASE RATE BY PREFERRED CHANNEL
# =========================================================

with right:

    st.subheader("📱 Purchase Rate by Preferred Channel")


    if "Preferred_Channel" in df.columns:

        channel = (
            df.groupby("Preferred_Channel")[target]
            .mean()
            .mul(100)
            .reset_index()
        )


        channel.columns = [
            "Preferred Channel",
            "Purchase Rate"
        ]


        fig_channel = px.bar(
            channel,
            x="Preferred Channel",
            y="Purchase Rate",
            text_auto=".2f"
        )


        fig_channel.update_traces(
            textposition="outside"
        )


        fig_channel.update_layout(
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            xaxis_title="Preferred Channel",
            yaxis_title="Purchase Rate (%)"
        )


        st.plotly_chart(
            fig_channel,
            use_container_width=True
        )


    else:

        st.warning(
            "Preferred_Channel column is not available."
        )


st.divider()


# =========================================================
# DASHBOARD SUMMARY
# =========================================================

st.header("💡 Dashboard Summary")

st.write(
    """
    This dashboard provides an overview of customer purchase
    behavior across different campaign and customer dimensions.

    The purchase distribution shows the overall proportion of
    customers who purchased and did not purchase.

    The customer segment, campaign type and preferred channel
    charts compare purchase rates across different groups.

    These descriptive insights can be used together with the
    machine learning model to support customer targeting and
    marketing decisions.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛍️ ShopSphere Campaign Response Analytics | "
    "Campaign Analytics Dashboard"
)