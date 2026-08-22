"""
Obesity Level Classification — Model Comparison Dashboard
============================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
# TABS FOR DETAILED SECTIONS
# ============================================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Predict", "Confusion Matrix", "Classification Report", "Feature Importance", "ROC Curves"
])

# ---- TAB 0: Live Prediction ----
with tab0:
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

# ---- TAB 1: Confusion Matrix ----
with tab1:
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Counts")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Row-wise Percentages")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ---- TAB 2: Classification Report ----
with tab2:
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

# ---- TAB 3: Feature Importance ----
with tab3:
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_, index=X_test.columns
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        top15 = importances.head(15)
        ax.barh(top15.index[::-1], top15.values[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances")
        st.pyplot(fig)

        st.dataframe(top15.reset_index().rename(
            columns={"index": "Feature", 0: "Importance"}
        ))
    else:
        st.info(f"{selected_model_name} does not expose feature_importances_.")

# ---- TAB 4: ROC Curves ----
with tab4:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = cycle(["#e6194B", "#3cb44b", "#4363d8", "#f58231",
                     "#911eb4", "#42d4f4", "#bfef45"])
    color_list = list(colors)[:n_classes]

    for i, color in zip(range(n_classes), color_list):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                      label=f"{class_names[i]} (AUC={roc_auc_i:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Full View")
    axes[0].legend(loc="lower right", fontsize=8)

    for i, color in zip(range(n_classes), color_list):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        axes[1].plot(fpr, tpr, color=color, lw=2)
    axes[1].plot([0, 0.3], [0, 0.3], "k--", lw=1)
    axes[1].set_xlim([0, 0.3])
    axes[1].set_ylim([0.7, 1.02])
    axes[1].set_title("Zoomed View (Low FPR)")

    st.pyplot(fig)

st.markdown("---")
st.caption("Obesity Levels dataset — UCI Machine Learning Repository")
