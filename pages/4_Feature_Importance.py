import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
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

st.title("🔍 Feature Importance")

BASE = Path(__file__).parents[1]
model = joblib.load(BASE / "models" / "xgboost_model.pkl")
feature_path = BASE / "models" / "feature_columns.pkl"

features = joblib.load(feature_path)
importance = model.feature_importances_

fi = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

top_n = st.slider("Number of features", 5, min(30, len(fi)), 10)

display = fi.head(top_n).sort_values("Importance")

fig = px.bar(
    display,
    x="Importance",
    y="Feature",
    orientation="h",
    text_auto=".3f",
    title="XGBoost Feature Importance",
    color_discrete_sequence=[
                "#F59E0B"
    ]
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(fi, use_container_width=True, hide_index=True)
