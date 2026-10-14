import streamlit as st
import pandas as pd
from pathlib import Path


# =========================================================
# PAGE TITLE
# =========================================================

st.title("💰 Business Metrics")

st.caption(
    "Campaign performance and marketing economics"
)
# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # -----------------------------------------------------
    # SHOPSPHERE LOGO / BRANDING
    # -----------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 10px 0 15px 0;
        ">
            <div style="font-size: 42px;">🛍️</div>
            <div style="
                font-size: 24px;
                font-weight: 700;
                margin-top: -5px;
            ">
                ShopSphere
            </div>
            <div style="
                font-size: 12px;
                opacity: 0.7;
            ">
                Campaign Response Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------------------
    # NAVIGATION
    # -----------------------------------------------------

    st.page_link(
        "app.py",
        label="🏠 App"
    )

    st.page_link(
        "pages/1_Dashboard.py",
        label="📊 Dashboard"
    )

    st.page_link(
        "pages/2_Model_Comparison.py",
        label="🤖 Model Comparison"
    )

    st.page_link(
        "pages/3_Customer_Prediction.py",
        label="🔮 Customer Prediction"
    )

    st.page_link(
        "pages/4_Feature_Importance.py",
        label="📈 Feature Importance"
    )

    st.page_link(
        "pages/5_Business_Metrics.py",
        label="💰 Business Metrics"
    )

    st.page_link(
        "pages/6_AI_Marketing_Assistant.py",
        label="🧠 AI Marketing Assistant"
    )

    st.divider()

    st.caption(
        "Customer Purchase Prediction\n\n"
        "ML + Business Intelligence"
    )



# =========================================================
# LOAD DATA
# =========================================================

BASE = Path(__file__).parents[1]

df = pd.read_csv(
    BASE / "data" / "campaign_customer_data.csv"
)


# =========================================================
# TARGET
# =========================================================

target = "Campaign_Purchased"


# =========================================================
# CALCULATE METRICS
# =========================================================

conversion_rate = df[target].mean() * 100

purchases = int(df[target].sum())


# Campaign Cost
if "Campaign_Cost" in df.columns:

    total_cost = df["Campaign_Cost"].sum()

else:

    total_cost = None


# Revenue
# The final notebook calculates revenue using Revenue_Per_Order.
if "Revenue_Per_Order" in df.columns:

    revenue = (
        df.loc[
            df[target] == 1,
            "Revenue_Per_Order"
        ].sum()
    )

else:

    revenue = None


# Cost per Acquisition
if total_cost is not None and purchases > 0:

    cpa = total_cost / purchases

else:

    cpa = None


# Marketing ROI
if (
    total_cost is not None
    and revenue is not None
    and total_cost > 0
):

    roi = (
        (revenue - total_cost)
        / total_cost
    ) * 100

else:

    roi = None


# =========================================================
# KPI ROW
# =========================================================

c1, c2, c3, c4 = st.columns(4)


# 1. Conversion Rate
with c1:

    st.metric(
        "📈 Conversion Rate",
        f"{conversion_rate:.2f}%"
    )


# 2. Purchases
with c2:

    st.metric(
        "🛒 Purchases",
        f"{purchases:,}"
    )


# 3. Campaign Cost
with c3:

    if total_cost is not None:

        st.metric(
            "💰 Campaign Cost",
            f"₹{total_cost:,.2f}"
        )

    else:

        st.metric(
            "💰 Campaign Cost",
            "N/A"
        )


# 4. Calculated Revenue
with c4:

    if revenue is not None:

        st.metric(
            "💵 Calculated Revenue",
            f"₹{revenue:,.2f}"
        )

    else:

        st.metric(
            "💵 Calculated Revenue",
            "N/A"
        )


# =========================================================
# SECOND KPI ROW
# =========================================================

st.write("")

c5, c6, c7, c8 = st.columns(4)


# 5. Cost per Acquisition
with c5:

    if cpa is not None:

        st.metric(
            "🎯 Cost per Acquisition",
            f"₹{cpa:,.2f}"
        )

    else:

        st.metric(
            "🎯 Cost per Acquisition",
            "N/A"
        )


# 6. Marketing ROI
with c6:

    if roi is not None:

        st.metric(
            "📊 Marketing ROI",
            f"{roi:,.2f}%"
        )

    else:

        st.metric(
            "📊 Marketing ROI",
            "N/A"
        )


# Keep remaining two columns empty for spacing
with c7:
    st.empty()

with c8:
    st.empty()


# =========================================================
# DIVIDER
# =========================================================

st.divider()


# =========================================================
# METRIC EXPLANATIONS
# =========================================================

st.subheader("📌 Metric Definitions")


col1, col2 = st.columns(2)


with col1:

    st.info(
        """
        **Cost per Acquisition (CPA)**

        CPA shows the average campaign cost for each
        customer who made a purchase.

        **Formula:**

        Campaign Cost ÷ Purchases
        """
    )


with col2:

    st.info(
        """
        **Marketing ROI**

        ROI compares the calculated revenue with the
        campaign cost.

        **Formula:**

        (Revenue − Campaign Cost) ÷ Campaign Cost × 100
        """
    )


# =========================================================
# NOTE
# =========================================================

st.warning(
    "The revenue and ROI definitions above reproduce the "
    "calculations in the final notebook. Review these definitions "
    "before presenting ROI as a real business result, because "
    "the notebook uses Revenue_Per_Order as revenue."
)