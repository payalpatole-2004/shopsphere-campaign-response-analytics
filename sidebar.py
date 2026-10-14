import streamlit as st


def show_sidebar():

    with st.sidebar:

        # -----------------------------
        # ShopSphere Branding
        # -----------------------------
        st.markdown(
            """
            <div style="text-align:center; padding:10px 0 15px 0;">
                <div style="font-size:42px;">🛍️</div>
                <div style="font-size:24px; font-weight:700;">
                    ShopSphere
                </div>
                <div style="font-size:12px; opacity:0.7;">
                    Campaign Response Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # -----------------------------
        # Navigation
        # -----------------------------
        st.page_link(
            "app.py",
            label="🏠 Home"
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

        st.caption("ML + Business Intelligence")