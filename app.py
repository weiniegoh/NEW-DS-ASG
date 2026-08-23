"""
Obesity Level Classification — Model Comparison Dashboard
============================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import gc

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
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
st.caption(
    "CRISP-DM Project | Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition"
)


# ============================================================
# MODEL REGISTRY
# ------------------------------------------------------------
# Add each new model here once it's trained & saved.
# ============================================================

MODEL_REGISTRY = {

    "Random Forest": {
        "model_path": "models/random_forest_model.pkl",
        "X_test_path": "data/X_test_encoded.pkl",
        "y_test_path": "data/y_test_flat.pkl",
        "y_pred_path": "data/rf_y_pred.pkl",
        "y_proba_path": "data/rf_y_pred_proba.pkl",

        "confusion_matrix_image": "images/rf_confusion_matrix.png",
        "feature_importance_image": "images/rf_feature_importance.png",
        "roc_curve_image": "images/rf_roc_curves.png",
    },

    # "Decision Tree": {
    #     "model_path": "models/decision_tree_model.pkl",
    #     "X_test_path": "data/X_test_encoded.pkl",
    #     "y_test_path": "data/y_test_flat.pkl",
    #     "y_pred_path": "data/dt_y_pred.pkl",
    #     "y_proba_path": "data/dt_y_pred_proba.pkl",
    #
    #     "confusion_matrix_image": "images/dt_confusion_matrix.png",
    #     "feature_importance_image": "images/dt_feature_importance.png",
    #     "roc_curve_image": "images/dt_roc_curves.png",
    # },

    # "Logistic Regression": {...},
    # "Gradient Boosting": {...},
}


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

@st.cache_resource
def load_artifacts(config):
    model = joblib.load(config["model_path"])
    X_test = joblib.load(config["X_test_path"])
    y_test = joblib.load(config["y_test_path"])
    y_pred = joblib.load(config["y_pred_path"])
    y_proba = joblib.load(config["y_proba_path"])

    return model, X_test, y_test, y_pred, y_proba


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

# ------------------------------------------------------------
# Model Selection
# ------------------------------------------------------------

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_REGISTRY.keys()),
    key="selected_model"
)

st.sidebar.markdown("---")


# ------------------------------------------------------------
# Prediction Inputs
# ------------------------------------------------------------

st.sidebar.subheader("Prediction Inputs")

st.sidebar.caption(
    "Enter an individual's information. "
    "The same inputs are automatically used when switching models."
)


# ------------------------------------------------------------
# Demographic & Physical Information
# ------------------------------------------------------------

st.sidebar.markdown("**Demographic & Physical Information**")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"],
    key="input_gender"
)

age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    max_value=100,
    value=25,
    key="input_age"
)

height = st.sidebar.number_input(
    "Height (m)",
    min_value=1.0,
    max_value=2.5,
    value=1.70,
    step=0.01,
    key="input_height"
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=250.0,
    value=70.0,
    step=0.5,
    key="input_weight"
)

family_history = st.sidebar.selectbox(
    "Family history of overweight?",
    ["yes", "no"],
    key="input_family_history"
)


# ------------------------------------------------------------
# Eating Habits
# ------------------------------------------------------------

st.sidebar.markdown("**Eating Habits**")

favc = st.sidebar.selectbox(
    "Frequent high-calorie food (FAVC)?",
    ["yes", "no"],
    key="input_favc"
)

fcvc = st.sidebar.slider(
    "Vegetable consumption frequency (FCVC)",
    1.0,
    3.0,
    2.0,
    step=0.1,
    key="input_fcvc"
)

ncp = st.sidebar.slider(
    "Number of main meals (NCP)",
    1.0,
    4.0,
    3.0,
    step=0.1,
    key="input_ncp"
)

caec = st.sidebar.selectbox(
    "Eating between meals (CAEC)",
    ["no", "Sometimes", "Frequently", "Always"],
    key="input_caec"
)

ch2o = st.sidebar.slider(
    "Daily water consumption (CH2O)",
    1.0,
    3.0,
    2.0,
    step=0.1,
    key="input_ch2o"
)

scc = st.sidebar.selectbox(
    "Monitors calorie consumption (SCC)?",
    ["no", "yes"],
    key="input_scc"
)

calc = st.sidebar.selectbox(
    "Alcohol consumption (CALC)",
    ["no", "Sometimes", "Frequently", "Always"],
    key="input_calc"
)


# ------------------------------------------------------------
# Lifestyle
# ------------------------------------------------------------

st.sidebar.markdown("**Lifestyle**")

smoke = st.sidebar.selectbox(
    "Smoker?",
    ["no", "yes"],
    key="input_smoke"
)

faf = st.sidebar.slider(
    "Physical activity frequency (FAF)",
    0.0,
    3.0,
    1.0,
    step=0.1,
    key="input_faf"
)

tue = st.sidebar.slider(
    "Time using technology devices (TUE)",
    0.0,
    2.0,
    1.0,
    step=0.1,
    key="input_tue"
)

mtrans = st.sidebar.selectbox(
    "Transportation used (MTRANS)",
    [
        "Automobile",
        "Motorbike",
        "Bike",
        "Public_Transportation",
        "Walking"
    ],
    key="input_mtrans"
)


# ============================================================
# LOAD SELECTED MODEL'S ARTIFACTS
# ============================================================

config = MODEL_REGISTRY[selected_model_name]

model, X_test, y_test, y_pred, y_proba = load_artifacts(config)

class_names = model.classes_
n_classes = len(class_names)

y_test_binarized = label_binarize(
    y_test,
    classes=class_names
)


# ============================================================
# LIVE PREDICTION
# ------------------------------------------------------------
# Prediction automatically updates whenever the sidebar
# inputs or selected model changes.
# ============================================================

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
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_columns,
    drop_first=True
)

input_aligned = input_encoded.reindex(
    columns=X_test.columns,
    fill_value=0
)

prediction = model.predict(input_aligned)[0]

prediction_proba = model.predict_proba(input_aligned)[0]

proba_df = pd.DataFrame({
    "Class": model.classes_,
    "Probability": prediction_proba
}).sort_values(
    "Probability",
    ascending=False
)


# ============================================================
# KEY METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

precision_w = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall_w = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

roc_auc_macro = roc_auc_score(
    y_test_binarized,
    y_proba,
    average="macro",
    multi_class="ovr"
)


# ============================================================
# MAIN AREA — PERFORMANCE OVERVIEW
# ============================================================

st.header(f"{selected_model_name} — Performance Overview")

st.caption(
    "Model performance and prediction results based on the "
    "currently selected model and prediction inputs."
)


# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "F1 (Weighted)",
    f"{f1_weighted:.4f}"
)

col3.metric(
    "Precision (Weighted)",
    f"{precision_w:.4f}"
)

col4.metric(
    "Recall (Weighted)",
    f"{recall_w:.4f}"
)

col5.metric(
    "ROC-AUC (Macro)",
    f"{roc_auc_macro:.4f}"
)


st.markdown("---")


# ============================================================
# PREDICTION RESULT
# ============================================================

st.subheader("Prediction Result")

prediction_col1, prediction_col2 = st.columns([1, 2])


with prediction_col1:

    st.markdown("### Predicted Obesity Level")

    st.success(
        f"## {prediction}"
    )

    st.caption(
        f"Prediction generated using the {selected_model_name} model."
    )


with prediction_col2:

    st.markdown("### Class Probabilities")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(
        proba_df["Class"][::-1],
        proba_df["Probability"][::-1],
        color="steelblue"
    )

    ax.set_xlabel("Predicted Probability")
    ax.set_xlim(0, 1)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    gc.collect()


st.markdown("---")


# ============================================================
# SECTION SELECTOR
# ============================================================

section = st.radio(
    "View Results",
    [
        "Confusion Matrix",
        "Classification Report",
        "Feature Importance",
        "ROC Curves"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("")


# ============================================================
# CONFUSION MATRIX
# ============================================================

if section == "Confusion Matrix":

    st.subheader("Confusion Matrix")

    st.caption(
        f"Confusion matrix for the {selected_model_name} model."
    )

    st.image(
        config["confusion_matrix_image"],
        width="stretch"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

elif section == "Classification Report":

    st.subheader("Classification Report")

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .round(4)
    )

    st.dataframe(
        report_df,
        width="stretch"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif section == "Feature Importance":

    st.subheader("Feature Importance")

    st.image(
        config["feature_importance_image"],
        width="stretch"
    )

    if hasattr(model, "feature_importances_"):

        importances = pd.Series(
            model.feature_importances_,
            index=X_test.columns
        ).sort_values(
            ascending=False
        )

        top15 = importances.head(15)

        feature_df = (
            top15
            .reset_index()
            .rename(
                columns={
                    "index": "Feature",
                    0: "Importance"
                }
            )
        )

        st.dataframe(
            feature_df,
            width="stretch"
        )


# ============================================================
# ROC CURVES
# ============================================================

elif section == "ROC Curves":

    st.subheader("ROC Curves")

    st.caption(
        f"One-vs-Rest ROC curves for the {selected_model_name} model."
    )

    st.image(
        config["roc_curve_image"],
        width="stretch"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Obesity Levels dataset — UCI Machine Learning Repository"
)
