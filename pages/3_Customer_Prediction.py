import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("🔮 Customer Purchase Prediction")
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

model_path = BASE / "models" / "xgboost_model.pkl"
feature_path = BASE / "models" / "feature_columns.pkl"
encoder_path = BASE / "models" / "label_encoders.pkl"

# Check required files
if not model_path.exists() or not feature_path.exists():
    st.warning("Run train_and_save.py first.")
    st.stop()

# Load files
model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)

if encoder_path.exists():
    encoders = joblib.load(encoder_path)
else:
    encoders = {}

st.info(
    "Enter customer and campaign information below. "
    "The prediction uses the XGBoost model and the same "
    "0.40 probability threshold used in the final notebook."
)

data = {}

# ---------------------------------------------------------
# Fallback categorical values
# These are only used if a saved encoder is not available.
# ---------------------------------------------------------

categorical = {
    "Gender": ["Male", "Female", "Other"],
    "Marital_Status": ["Single", "Married", "Divorced", "Unknown"],
    "Income_Bracket": ["Low", "Medium", "High", "Unknown"],
    "Occupation": [
        "Student",
        "Employed",
        "Self-employed",
        "Retired",
        "Unknown"
    ],
    "Customer_Segment": [
        "Premium",
        "Regular",
        "Basic"
    ],
    "Loyalty_Tier": [
        "Bronze",
        "Silver",
        "Gold",
        "Platinum"
    ],
    "Preferred_Channel": [
        "Email",
        "SMS",
        "App",
        "Social Media"
    ],
    "Campaign_Type": [
        "Email",
        "SMS",
        "App",
        "Social Media"
    ],
    "Device_Type": [
        "Mobile",
        "Desktop",
        "Tablet"
    ],
    "Traffic_Source": [
        "Organic",
        "Paid",
        "Social",
        "Direct",
        "Referral"
    ]
}

# ---------------------------------------------------------
# Numeric default values
# ---------------------------------------------------------

numeric_defaults = {
    "Age": 30,
    "Annual_Income": 500000.0,
    "Credit_Score": 700,
    "Household_Size": 3,
    "Tenure_Months": 24,
    "Last_Login_Days_Ago": 10,
    "Total_Orders": 10,
    "Avg_Order_Value": 10000.0,
    "Total_Quantity": 20,
    "Total_Tickets": 2,
    "Avg_Satisfaction": 4.0,
    "Escalation_Rate": 0.1,
    "Campaigns_Received": 5,
    "Campaign_Open_Rate": 0.25,
    "Campaign_Click_Rate": 0.10,
    "Total_Sessions": 20,
    "Avg_Pages_Viewed": 5.0,
    "Cart_Rate": 0.5,
    "Customer_Lifetime_Value": 100000.0,
    "Campaign_Cost": 10.0,
    "Session_Duration": 300.0,
    "Added_To_Cart": 0.2,
    "Resolution_Time_Hours": 12.0,
    "Escalated": 0
}

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.subheader("Customer & Campaign Information")

cols = st.columns(3)

for idx, col_name in enumerate(feature_columns):

    container = cols[idx % 3]

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # Use EXACT categories from saved encoder
    # -----------------------------------------------------

    if col_name in encoders:

        encoder = encoders[col_name]

        options = encoder.classes_.tolist()

        selected_value = container.selectbox(
            col_name,
            options=options,
            format_func=lambda x: str(x)
        )

        data[col_name] = selected_value

    # -----------------------------------------------------
    # FALLBACK CATEGORICAL FEATURES
    # -----------------------------------------------------

    elif col_name in categorical:

        data[col_name] = container.selectbox(
            col_name,
            categorical[col_name]
        )

    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    elif col_name in numeric_defaults:

        default = numeric_defaults[col_name]

        if isinstance(default, int):

            data[col_name] = container.number_input(
                col_name,
                value=default
            )

        else:

            data[col_name] = container.number_input(
                col_name,
                value=float(default)
            )

    # -----------------------------------------------------
    # OTHER NUMERIC FEATURES
    # -----------------------------------------------------

    else:

        data[col_name] = container.number_input(
            col_name,
            value=0.0
        )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button(
    "🚀 Predict Campaign Purchase",
    type="primary"
):

    row = pd.DataFrame([data])

    # -----------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # -----------------------------------------------------

    for col, encoder in encoders.items():

        if col in row.columns:

            try:

                # First try using original data type
                row[col] = encoder.transform(
                    row[col]
                )

            except Exception:

                try:

                    # If encoder was trained using strings
                    row[col] = encoder.transform(
                        row[col].astype(str)
                    )

                except Exception as exc:

                    st.error(
                        f"Encoding error in {col}: {exc}"
                    )

                    st.stop()

    # -----------------------------------------------------
    # Match exact feature order used during training
    # -----------------------------------------------------

    row = row.reindex(
        columns=feature_columns
    )

    # -----------------------------------------------------
    # XGBoost prediction
    # -----------------------------------------------------

    try:

        probability = float(
            model.predict_proba(row)[:, 1][0]
        )

        prediction = int(
            probability >= 0.40
        )

    except Exception as exc:

        st.error(
            f"Prediction error: {exc}"
        )

        st.stop()

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )

    with c2:

        if prediction == 1:

            st.success(
                "🟢 Likely to Purchase"
            )

        else:

            st.warning(
                "🟡 Less Likely to Purchase"
            )

    st.progress(
        min(probability, 1.0)
    )

    st.caption(
        "Prediction threshold: 0.40 "
        "(same threshold used in the final notebook)."
    )