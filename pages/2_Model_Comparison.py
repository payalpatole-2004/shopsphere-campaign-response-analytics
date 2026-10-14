import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.title("🤖 Model Comparison")
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


BASE = Path(__file__).parents[1]
metrics_path = BASE / "models" / "model_metrics.csv"

if not metrics_path.exists():
    st.warning("Run train_and_save.py first.")
    st.stop()

metrics = pd.read_csv(metrics_path)

st.subheader("Evaluation Metrics")

st.dataframe(
    metrics.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "ROC_AUC": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

metric = st.selectbox(
    "Select metric to compare",
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
)

fig = px.bar(
    metrics,
    x="Model",
    y=metric,
    text_auto=".3f",
    title=f"{metric} Comparison"
)
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Confusion Matrix")

model_name = st.selectbox(
    "Select model",
    metrics["Model"].tolist()
)

cm_path = BASE / "models" / f"{model_name.lower().replace(' ', '_')}_cm.csv"

if cm_path.exists():
    cm = pd.read_csv(cm_path, index_col=0)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["0 - Not Purchased", "1 - Purchased"],
        y=["0 - Not Purchased", "1 - Purchased"],
        title=f"{model_name} Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Confusion matrix file not available for this model.")
