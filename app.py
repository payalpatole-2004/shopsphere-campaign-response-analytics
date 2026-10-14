import streamlit as st
from sidebar import show_sidebar


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ShopSphere | Home",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

show_sidebar()


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🛍️ ShopSphere Campaign Response Analytics")

st.subheader(
    "Customer Purchase Prediction & Marketing Decision Support"
)

st.write(
    "An end-to-end analytics project that combines customer, "
    "campaign, website and customer service data to predict "
    "campaign purchase response and support better marketing decisions."
)


st.divider()


# =========================================================
# BUSINESS PROBLEM
# =========================================================

st.header("🎯 Business Problem")

with st.container(border=True):

    st.write(
        """
        Marketing teams send campaigns to many customers, but
        every customer does not have the same probability of making
        a purchase.

        The business needs a way to identify customers who are more
        likely to purchase so that marketing campaigns can be
        targeted more effectively.
        """
    )


# =========================================================
# PROJECT OBJECTIVE
# =========================================================

st.header("💡 Project Objective")

with st.container(border=True):

    st.write(
        """
        The main objective of this project is to predict whether a
        customer is likely to purchase after receiving a marketing
        campaign.

        The project combines customer information, campaign
        engagement, website activity and customer service data
        and applies machine learning classification models.
        """
    )


# =========================================================
# PROJECT WORKFLOW
# =========================================================

st.header("🔄 Project Workflow")

workflow = [
    "📁 Customer Data",
    "🧹 Data Cleaning & Preprocessing",
    "⚙️ Feature Engineering",
    "✂️ Train / Test Split",
    "🤖 Model Training",
    "📊 Model Evaluation",
    "🎯 Probability / Threshold Tuning",
    "🔮 Customer Purchase Prediction",
    "💡 Marketing Insights"
]

for i, step in enumerate(workflow):

    if i < len(workflow) - 1:
        st.write(f"**{i + 1}. {step}** →")
    else:
        st.write(f"**{i + 1}. {step}**")


st.divider()


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.header("📌 Project Summary")

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
        **Business Problem**

        Marketing teams need to identify customers who are
        more likely to purchase after receiving campaigns.

        This can help support better customer targeting.
        """
    )


with col2:

    st.success(
        """
        **Analytics Solution**

        Customer, campaign, website and service data are
        combined and transformed.

        Classification models are then used to predict
        campaign purchase response.
        """
    )


st.divider()


# =========================================================
# PROJECT COMPONENTS
# =========================================================

st.header("🚀 Project Components")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 📊")
    st.write("**Business Dashboard**")
    st.caption("Monitor campaign and customer analytics.")

with c2:
    st.markdown("### 🤖")
    st.write("**Machine Learning**")
    st.caption("Predict customer purchase response.")

with c3:
    st.markdown("### 🔮")
    st.write("**Prediction System**")
    st.caption("Predict purchase probability for a customer.")

with c4:
    st.markdown("### 🧠")
    st.write("**AI Assistant**")
    st.caption("Explain results in simple business language.")


st.divider()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🛍️ ShopSphere Campaign Response Analytics | "
    "Customer Purchase Prediction & Marketing Decision Support"
)