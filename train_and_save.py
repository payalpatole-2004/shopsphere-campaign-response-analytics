"""
Run this file from the project root after placing the original CSV files
in the data/ folder.

This reproduces the final notebook's preprocessing and trains/saves the
four model approaches for Streamlit.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

customers = pd.read_csv(DATA / "Customers.csv")[[
    "Customer_ID","Age","Gender","Marital_Status","Income_Bracket",
    "Annual_Income","Occupation","Customer_Segment","Loyalty_Tier",
    "Preferred_Channel","App_User_Flag","Credit_Score","Household_Size",
    "Tenure_Months","Last_Login_Days_Ago"
]]

CDT = pd.read_csv(DATA / "Customer_Department_Targets - Copy.csv")[[
    "Customer_ID","Total_Orders","Avg_Order_Value","Total_Quantity",
    "Recency_Days","Total_Tickets","Avg_Satisfaction","Escalation_Rate",
    "Campaigns_Received","Campaign_Open_Rate","Campaign_Click_Rate",
    "Total_Sessions","Avg_Pages_Viewed","Cart_Rate",
    "Customer_Lifetime_Value"
]]

MC = pd.read_csv(DATA / "Marketing_Campaigns.csv")[[
    "Customer_ID","Campaign_Type","Campaign_Cost","Purchased"
]]

WA = pd.read_csv(DATA / "Website_Activity.csv")[[
    "Customer_ID","Device_Type","Traffic_Source","Session_Duration",
    "Added_To_Cart"
]]

CS = pd.read_csv(DATA / "Customer_Service.csv")[[
    "Customer_ID","Resolution_Time_Hours","Escalated"
]]

target = (
    MC.groupby("Customer_ID")["Purchased"]
    .max()
    .reset_index()
    .rename(columns={"Purchased": "Campaign_Purchased"})
)

website_features = (
    WA.groupby("Customer_ID")
    .agg({"Session_Duration": "mean", "Added_To_Cart": "mean"})
    .reset_index()
)

service_features = (
    CS.groupby("Customer_ID")
    .agg({"Resolution_Time_Hours": "mean", "Escalated": "mean"})
    .reset_index()
)

df = customers.merge(CDT, on="Customer_ID", how="left")
df = df.merge(MC, on="Customer_ID", how="left")
df = df.merge(website_features, on="Customer_ID", how="left")
df = df.merge(service_features, on="Customer_ID", how="left")
df = df.merge(target, on="Customer_ID", how="left")

df["Income_Bracket"] = df["Income_Bracket"].fillna("Unknown")
df["Occupation"] = df["Occupation"].fillna("Unknown")

df["Annual_Income"] = df.groupby(
    "Income_Bracket"
)["Annual_Income"].transform(
    lambda x: x.fillna(x.median())
)

df = df[df["Campaign_Purchased"].notna()].copy()

df[["Session_Duration","Added_To_Cart"]] = df[
    ["Session_Duration","Added_To_Cart"]
].fillna(0)

df["Escalated"] = df["Escalated"].fillna(0).astype(int)
df["Resolution_Time_Hours"] = df["Resolution_Time_Hours"].fillna(0)

df["Revenue_Per_Order"] = (
    df["Customer_Lifetime_Value"] / (df["Total_Orders"] + 1)
)

df["Revenue_Per_Session"] = (
    df["Customer_Lifetime_Value"] / (df["Total_Sessions"] + 1)
)

df["Loyalty_Score"] = (
    df["Tenure_Months"] +
    df["Total_Orders"] +
    df["Campaigns_Received"]
)

df["Risk_Score"] = (
    df["Last_Login_Days_Ago"] +
    df["Total_Tickets"] * 5 +
    df["Escalation_Rate"] * 100
)

# Keep the original business-facing dataset for Streamlit.
df.to_csv(DATA / "campaign_customer_data.csv", index=False)

cat_cols = df.select_dtypes(include="object").columns.tolist()

encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

df = df.drop(columns=["Customer_ID", "Purchased", "Recency_Days"])
df.drop_duplicates(inplace=True)

X = df.drop(columns=["Campaign_Purchased"])
y = df["Campaign_Purchased"]

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, MODELS / "feature_columns.pkl")
joblib.dump(encoders, MODELS / "label_encoders.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODELS / "scaler.pkl")

def evaluate(name, model, Xtr, Xte, threshold=None):
    model.fit(Xtr, y_train)

    if threshold is None:
        pred = model.predict(Xte)
    else:
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= threshold).astype(int)

    prob = model.predict_proba(Xte)[:, 1]

    cm = confusion_matrix(y_test, pred)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob)
    }

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    cm_df.to_csv(
        MODELS / f"{name.lower().replace(' ', '_')}_cm.csv"
    )

    joblib.dump(model, MODELS / f"{name.lower().replace(' ', '_')}_model.pkl")

    return metrics

results = []

# Logistic Regression: SMOTE version from the final notebook.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, y_train
)

log_model = LogisticRegression()
log_model.fit(X_train_smote, y_train_smote)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

log_metrics = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, log_pred),
    "Precision": precision_score(y_test, log_pred, zero_division=0),
    "Recall": recall_score(y_test, log_pred, zero_division=0),
    "F1": f1_score(y_test, log_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, log_prob)
}

pd.DataFrame(
    confusion_matrix(y_test, log_pred),
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
).to_csv(MODELS / "logistic_regression_cm.csv")

joblib.dump(log_model, MODELS / "logistic_regression_model.pkl")
results.append(log_metrics)

cart = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
results.append(
    evaluate(
        "Decision Tree",
        cart,
        X_train,
        X_test
    )
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
results.append(
    evaluate(
        "Random Forest",
        rf,
        X_train,
        X_test
    )
)

xgb = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    learning_rate=0.2,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=1.0,
    eval_metric="logloss",
    random_state=42
)

results.append(
    evaluate(
        "XGBoost",
        xgb,
        X_train,
        X_test,
        threshold=0.40
    )
)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(MODELS / "model_metrics.csv", index=False)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    xgb,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

pd.DataFrame({
    "Fold": range(1, 6),
    "Accuracy": cv_scores
}).to_csv(MODELS / "xgb_cv_scores.csv", index=False)

print("\nModels saved successfully.")
print(metrics_df)
print("\nXGBoost CV mean accuracy:", cv_scores.mean())
