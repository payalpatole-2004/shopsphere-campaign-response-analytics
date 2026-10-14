import streamlit as st
import pandas as pd
from pathlib import Path
from sidebar import show_sidebar

st.title("🤖 AI Marketing Assistant")
show_sidebar()
st.caption(
    "A free local marketing assistant that interprets model metrics "
    "and provides business-oriented explanations."
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE = Path(__file__).parents[1]
metrics_path = BASE / "models" / "model_metrics.csv"

# ---------------------------------------------------------
# LOAD MODEL METRICS
# ---------------------------------------------------------

if not metrics_path.exists():
    st.warning("Run train_and_save.py first.")
    st.stop()

metrics = pd.read_csv(metrics_path)

# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

st.subheader("📊 Model Context")

model_name = st.selectbox(
    "Select model",
    metrics["Model"].tolist()
)

selected = metrics[
    metrics["Model"] == model_name
].iloc[0]

# ---------------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------------

st.dataframe(
    pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ],
        "Value": [
            selected["Accuracy"],
            selected["Precision"],
            selected["Recall"],
            selected["F1"],
            selected["ROC_AUC"]
        ]
    }),
    hide_index=True,
    use_container_width=True
)

# ---------------------------------------------------------
# QUESTION
# ---------------------------------------------------------

st.subheader("💬 Ask the Marketing Assistant")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: Explain the model performance in simple business language."
    )
)

# ---------------------------------------------------------
# LOCAL RESPONSE FUNCTION
# ---------------------------------------------------------

def generate_local_insight(question, selected, model_name):

    q = question.lower()

    accuracy = float(selected["Accuracy"])
    precision = float(selected["Precision"])
    recall = float(selected["Recall"])
    f1 = float(selected["F1"])
    roc_auc = float(selected["ROC_AUC"])

    # Convert to percentage
    accuracy_pct = accuracy * 100
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100
    roc_auc_pct = roc_auc * 100

    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    if (
        "performance" in q
        or "perform" in q
        or "overall" in q
    ):
        return f"""
### 📊 {model_name} Model Performance

The **{model_name}** model has:

- **Accuracy:** {accuracy_pct:.2f}%
- **Precision:** {precision_pct:.2f}%
- **Recall:** {recall_pct:.2f}%
- **F1 Score:** {f1_pct:.2f}%
- **ROC-AUC:** {roc_auc_pct:.2f}%

### Business meaning

The model is being evaluated on how well it identifies customers
who are likely to purchase after a marketing campaign.

**Accuracy** shows the overall percentage of correct predictions.

**Precision** tells us how reliable the model's positive predictions
are. A higher precision means fewer customers are incorrectly
identified as potential purchasers.

**Recall** tells us how many of the actual purchasing customers
were successfully identified.

**F1 Score** balances precision and recall.

**ROC-AUC** indicates how well the model separates purchasing
customers from non-purchasing customers.

For marketing decisions, it is important to look beyond accuracy
and consider precision, recall and F1 score as well.
"""

    # -----------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------

    elif "accuracy" in q:

        return f"""
### 🎯 Accuracy

The **{model_name}** model has an accuracy of **{accuracy_pct:.2f}%**.

Accuracy tells us how many predictions were correct out of all
predictions.

For example, if accuracy is 90%, approximately 90 out of every
100 predictions were classified correctly.

However, accuracy alone should not be used to judge a model,
especially when the purchasing and non-purchasing classes are
imbalanced.
"""

    # -----------------------------------------------------
    # PRECISION
    # -----------------------------------------------------

    elif "precision" in q:

        return f"""
### 🎯 Precision

The **{model_name}** model has a precision of **{precision_pct:.2f}%**.

Precision answers this question:

**"When the model predicts that a customer will purchase,
how often is that prediction correct?"**

A higher precision means the marketing team is less likely to
target customers who are unlikely to purchase.

This can help reduce unnecessary campaign spending.
"""

    # -----------------------------------------------------
    # RECALL
    # -----------------------------------------------------

    elif "recall" in q:

        return f"""
### 🔎 Recall

The **{model_name}** model has a recall of **{recall_pct:.2f}%**.

Recall answers this question:

**"Out of all customers who actually purchased,
how many did the model successfully identify?"**

A low recall means the model is missing some potential purchasers.

For a marketing campaign, this could mean some customers who
were actually interested in purchasing may not be targeted.
"""

    # -----------------------------------------------------
    # F1
    # -----------------------------------------------------

    elif (
        "f1" in q
        or "f1 score" in q
        or "f1-score" in q
    ):

        return f"""
### ⚖️ F1 Score

The **{model_name}** model has an F1 score of **{f1_pct:.2f}%**.

F1 Score combines **precision and recall** into one metric.

It is useful when we want a balance between:

- Correct positive predictions
- Finding actual positive customers

A higher F1 score generally means the model has a better balance
between precision and recall.
"""

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    elif (
        "roc" in q
        or "auc" in q
        or "roc-auc" in q
    ):

        return f"""
### 📈 ROC-AUC

The **{model_name}** model has a ROC-AUC of **{roc_auc:.4f}**
or approximately **{roc_auc_pct:.2f}%**.

ROC-AUC measures how well the model can distinguish between
customers who purchase and customers who do not purchase.

A value closer to **1.0** indicates stronger separation,
while a value around **0.5** indicates performance similar to
random classification.
"""

    # -----------------------------------------------------
    # MARKETING RECOMMENDATION
    # -----------------------------------------------------

    elif (
        "marketing" in q
        or "recommendation" in q
        or "recommend" in q
        or "business" in q
        or "improve" in q
        or "target" in q
    ):

        recommendation = ""

        if recall < 0.70:
            recommendation += """
- The recall is relatively low, so the marketing team should
  be aware that some actual purchasers may be missed.
"""

        if precision >= 0.80:
            recommendation += """
- The precision is relatively high, so positive predictions can
  be useful when the campaign budget needs to be controlled.
"""

        if roc_auc >= 0.80:
            recommendation += """
- The ROC-AUC indicates that the model has good ability to
  distinguish purchasers from non-purchasers.
"""

        if not recommendation:
            recommendation = """
- Review precision and recall together before using the model
  for customer targeting.
"""

        return f"""
### 💡 Marketing Interpretation

For the **{model_name}** model:

- Accuracy: **{accuracy_pct:.2f}%**
- Precision: **{precision_pct:.2f}%**
- Recall: **{recall_pct:.2f}%**
- F1 Score: **{f1_pct:.2f}%**
- ROC-AUC: **{roc_auc_pct:.2f}%**

### Suggested business considerations

{recommendation}

These are recommendations based only on the model evaluation
metrics. They do not represent measured business impact.
"""

    # -----------------------------------------------------
    # SIMPLE EXPLANATION
    # -----------------------------------------------------

    elif (
        "explain" in q
        or "simple" in q
        or "meaning" in q
    ):

        return f"""
### 🧠 Simple Explanation

The **{model_name}** model is being used to predict whether
a customer is likely to purchase after a marketing campaign.

Its current results are:

**Accuracy:** {accuracy_pct:.2f}%

**Precision:** {precision_pct:.2f}%

**Recall:** {recall_pct:.2f}%

**F1 Score:** {f1_pct:.2f}%

**ROC-AUC:** {roc_auc_pct:.2f}%

In simple terms, these metrics tell us how accurately the model
makes predictions and how well it identifies customers who
actually purchase.

The marketing team can use these results to understand the
strengths and limitations of the prediction model.
"""

    # -----------------------------------------------------
    # DEFAULT RESPONSE
    # -----------------------------------------------------

    else:

        return f"""
### 🤖 Local Marketing Assistant

I can explain the **{model_name}** model using the available
evaluation metrics.

Try asking:

- "Explain the model performance"
- "What does accuracy mean?"
- "Explain precision"
- "What does recall mean?"
- "Explain F1 score"
- "What does ROC-AUC mean?"
- "Give me a marketing recommendation"
- "How can the marketing team use this model?"

Current model:

**{model_name}**

Accuracy: **{accuracy_pct:.2f}%**

Precision: **{precision_pct:.2f}%**

Recall: **{recall_pct:.2f}%**

F1 Score: **{f1_pct:.2f}%**

ROC-AUC: **{roc_auc_pct:.2f}%**
"""


# ---------------------------------------------------------
# GENERATE INSIGHT
# ---------------------------------------------------------

if st.button("✨ Generate AI Insight", type="primary"):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        insight = generate_local_insight(
            question,
            selected,
            model_name
        )

        st.subheader("💡 Marketing Insight")

        st.write(insight)