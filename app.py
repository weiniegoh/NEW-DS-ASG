"""
Obesity Level Classification — Model Comparison Dashboard
============================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, lower memory overhead on servers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import gc
from itertools import cycle

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Obesity Level Classification Dashboard",
    layout="wide"
)

st.title("Obesity Level Classification — Model Comparison")
st.caption("CRISP-DM Project | Estimation of Obesity Levels Based on Eating Habits and Physical Condition")

# ============================================================
# MODEL REGISTRY
# ------------------------------------------------------------
# Add each new model here once it's trained & saved.
# Every model must have: model file, X_test, y_test, y_pred, y_pred_proba
# ============================================================
MODEL_REGISTRY = {
    "Random Forest": {
        "model_path": "models/random_forest_model.pkl",
        "X_test_path": "data/X_test_encoded.pkl",
        "y_test_path": "data/y_test_flat.pkl",
        "y_pred_path": "data/rf_y_pred.pkl",
        "y_proba_path": "data/rf_y_pred_proba.pkl",
    },
    # "Decision Tree": {
    #     "model_path": "models/decision_tree_model.pkl",
    #     "X_test_path": "data/X_test_encoded.pkl",
    #     "y_test_path": "data/y_test_flat.pkl",
    #     "y_pred_path": "data/dt_y_pred.pkl",
    #     "y_proba_path": "data/dt_y_pred_proba.pkl",
    # },
    # "Logistic Regression": {...},
    # "Gradient Boosting": {...},
}

# ============================================================
# SIDEBAR — MODEL SELECTION
# ============================================================
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model to view results:",
    list(MODEL_REGISTRY.keys())
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Select a model above to view its accuracy, F1-score, ROC-AUC, "
    "confusion matrix, and feature importance."
)

# ============================================================
# LOAD SELECTED MODEL'S ARTIFACTS (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_artifacts(config):
    model = joblib.load(config["model_path"])
    X_test = joblib.load(config["X_test_path"])
    y_test = joblib.load(config["y_test_path"])
    y_pred = joblib.load(config["y_pred_path"])
    y_proba = joblib.load(config["y_proba_path"])
    return model, X_test, y_test, y_pred, y_proba

config = MODEL_REGISTRY[selected_model_name]
model, X_test, y_test, y_pred, y_proba = load_artifacts(config)

class_names = model.classes_
n_classes = len(class_names)
y_test_binarized = label_binarize(y_test, classes=class_names)

# ============================================================
# KEY METRICS
# ============================================================
st.header(f"{selected_model_name} — Performance Overview")

accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
precision_w = precision_score(y_test, y_pred, average="weighted")
recall_w = recall_score(y_test, y_pred, average="weighted")
roc_auc_macro = roc_auc_score(y_test_binarized, y_proba, average="macro", multi_class="ovr")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("F1 (Weighted)", f"{f1_weighted:.4f}")
col3.metric("Precision (Weighted)", f"{precision_w:.4f}")
col4.metric("Recall (Weighted)", f"{recall_w:.4f}")
col5.metric("ROC-AUC (Macro)", f"{roc_auc_macro:.4f}")

st.markdown("---")

# ============================================================
# SECTION SELECTOR (radio instead of tabs — only the selected
# section's code executes, which keeps memory usage low)
# ============================================================
section = st.radio(
    "View:",
    ["Predict", "Confusion Matrix", "Classification Report", "Feature Importance", "ROC Curves"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("")

# ---- SECTION: Live Prediction ----
if section == "Predict":
    st.subheader("Predict Obesity Level")
    st.caption("Enter an individual's information below to get a predicted obesity category.")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)
        family_history = st.selectbox("Family history of overweight?", ["yes", "no"])

    with c2:
        favc = st.selectbox("Frequent high-calorie food (FAVC)?", ["yes", "no"])
        fcvc = st.slider("Vegetable consumption frequency (FCVC)", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("Eating between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Smoker?", ["no", "yes"])

    with c3:
        ch2o = st.slider("Daily water consumption (CH2O)", 1.0, 3.0, 2.0, step=0.1)
        scc = st.selectbox("Monitors calorie consumption (SCC)?", ["no", "yes"])
        faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Time using technology devices (TUE)", 0.0, 2.0, 1.0, step=0.1)
        calc = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox(
            "Transportation used (MTRANS)",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
        )

    if st.button("Predict Obesity Level", type="primary"):
        # Build a single-row DataFrame matching the ORIGINAL (pre-encoding) column structure
        input_dict = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
        }
        input_df = pd.DataFrame([input_dict])

        categorical_columns = [
            "Gender", "family_history_with_overweight", "FAVC",
            "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
        ]

        # One-hot encode the single input row the same way training data was encoded
        input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

        # Align to the EXACT columns the model was trained on (fills missing dummy
        # columns with 0, drops any that don't belong, and puts them in the right order)
        input_aligned = input_encoded.reindex(columns=X_test.columns, fill_value=0)

        prediction = model.predict(input_aligned)[0]
        prediction_proba = model.predict_proba(input_aligned)[0]

        st.success(f"**Predicted Obesity Level: {prediction}**")

        proba_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": prediction_proba
        }).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(proba_df["Class"][::-1], proba_df["Probability"][::-1], color="steelblue")
        ax.set_xlabel("Predicted Probability")
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

# ---- SECTION: Confusion Matrix ----
elif section == "Confusion Matrix":
    st.image("images/rf_confusion_matrix.png", width="stretch")

# ---- SECTION: Classification Report ----
elif section == "Classification Report":
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(report_df, width="stretch")

# ---- SECTION: Feature Importance ----
elif section == "Feature Importance":
    st.image("images/rf_feature_importance.png", width="stretch")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=X_test.columns
        ).sort_values(ascending=False)
        top15 = importances.head(15)
        st.dataframe(top15.reset_index().rename(
            columns={"index": "Feature", 0: "Importance"}
        ))

# ---- SECTION: ROC Curves ----
elif section == "ROC Curves":
    st.image("images/rf_roc_curves.png", width="stretch")

st.markdown("---")
st.caption("Obesity Levels dataset — UCI Machine Learning Repository")
