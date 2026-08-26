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
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ============================================================
# XGBOOST CLASS ORDER
# ============================================================
# XGBClassifier is trained with integer target IDs (0..6).
# The exported xgb_y_pred.pkl uses the original string labels.
# Keep this order identical to LabelEncoder.classes_ in XGBoost.ipynb.
XGB_CLASS_NAMES = np.array([
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II"
], dtype=str)


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
# ============================================================
# Add each new model here once its files are available.
#
# For models that are not ready yet, keep:
#     "available": False
#
# Once the model is ready, change it to:
#     "available": True
#
# and provide the required file paths.
# ============================================================

MODEL_REGISTRY = {

    "Logistic Regression": {
        "available": True,

        "model_path": "models/logistic_regression_model.pkl",
        "X_test_path": "data/X_test_encoded.pkl",
        "y_test_path": "data/y_test_flat.pkl",
        "y_pred_path": "data/lr_y_pred.pkl",
        "y_proba_path": "data/lr_y_pred_proba.pkl",

        "confusion_matrix_image": "images/lr_confusion_matrix.png",
        "feature_importance_image": "images/lr_feature_importance.png",
        "roc_curve_image": "images/lr_roc_curves.png",
        "feature_importance_data":
        "data/lr_feature_importance.csv",
        
    },
    
    "Random Forest": {
        "available": True,

        "model_path": "models/random_forest_model.pkl",
        "X_test_path": "data/X_test_encoded.pkl",
        "y_test_path": "data/y_test_flat.pkl",
        "y_pred_path": "data/rf_y_pred.pkl",
        "y_proba_path": "data/rf_y_pred_proba.pkl",

        "confusion_matrix_image":
            "images/rf_confusion_matrix.png",

        "feature_importance_image":
            "images/rf_feature_importance.png",

        "roc_curve_image":
            "images/rf_roc_curves.png",
    },

"K-Nearest Neighbours (KNN)": {
    "available": True,
    "model_path": "models/knn_model.pkl",
    "X_test_path": "data/X_test_encoded.pkl",
    "y_test_path": "data/y_test_flat.pkl",
    "y_pred_path": "data/knn_y_pred.pkl",
    "y_proba_path": "data/knn_y_pred_proba.pkl",
    "confusion_matrix_image": "images/knn_confusion_matrix.png",
    "feature_importance_image": "images/knn_feature_importance.png",
    "roc_curve_image": "images/knn_roc_curves.png",
     "feature_importance_data": "data/knn_feature_importance.csv",
},

    "XGBoost": {
        "available": True,
        "model_path": "models/xgboost_model.pkl",
        "X_test_path": "data/X_test_encoded.pkl",
        "y_test_path": "data/y_test_flat.pkl",
        "y_pred_path": "data/xgb_y_pred.pkl",
        "y_proba_path": "data/xgb_y_pred_proba.pkl",
        "confusion_matrix_image": "images/xgb_confusion_matrix.png",
        "feature_importance_image": "images/xgb_feature_importance.png",
        "roc_curve_image": "images/xgb_roc_curves.png",
    },
}


# ============================================================
# LOAD MODEL ARTIFACTS
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


# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.subheader("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_REGISTRY.keys()),
    key="selected_model"
)

selected_config = MODEL_REGISTRY[selected_model_name]


# ============================================================
# PREDICTION INPUTS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Inputs")

st.sidebar.caption(
    "Enter an individual's information. "
    "These inputs remain unchanged when switching models."
)


# ------------------------------------------------------------
# DEMOGRAPHIC & PHYSICAL INFORMATION
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
# BMI
# ------------------------------------------------------------

bmi = weight / (height ** 2)

st.sidebar.metric(
    "Calculated BMI",
    f"{bmi:.2f}"
)


# ------------------------------------------------------------
# EATING HABITS
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
# LIFESTYLE
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
# CHECK WHETHER MODEL IS AVAILABLE
# ============================================================

if not selected_config.get("available", False):

    st.info(
        f"**{selected_model_name}** has not been imported yet. "
        "The model will appear here once its trained model and "
        "prediction artifacts are added."
    )

    # --------------------------------------------------------
    # MODEL COMPARISON TABLE STILL SHOWN
    # --------------------------------------------------------

    st.header("Model Comparison")




# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model, X_test, y_test, y_pred, y_proba = load_artifacts(
    selected_config
)

# Keep evaluation labels consistent across every model.
y_test = np.asarray(y_test).ravel().astype(str)
y_pred = np.asarray(y_pred).ravel().astype(str)

# A plain XGBClassifier stores integer classes (0..6), while the shared
# y_test and exported XGBoost predictions are obesity-level strings.
# Use the fixed LabelEncoder class order only for XGBoost.
if selected_model_name == "XGBoost":
    class_names = XGB_CLASS_NAMES
else:
    class_names = np.asarray(model.classes_).astype(str)

y_test_binarized = label_binarize(
    y_test,
    classes=class_names
)


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

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
# PREDICTION
# ============================================================

input_dict = {

    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,

    "family_history_with_overweight":
        family_history,

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


# Build the encoded row manually — pd.get_dummies is unreliable on a
# single row because drop_first drops whatever category is present,
# regardless of which category it actually is.
#
# IMPORTANT FOR XGBOOST:
# XGBoost validates feature NAMES as well as the number/order of features.
# Therefore, when XGBoost is selected, build the input row from the exact
# feature names stored inside the fitted booster instead of blindly using
# the shared X_test columns.
if selected_model_name == "XGBoost":
    prediction_columns = model.get_booster().feature_names

    if not prediction_columns:
        prediction_columns = X_test.columns.tolist()
else:
    prediction_columns = X_test.columns.tolist()

input_aligned = pd.DataFrame(
    0.0,
    index=[0],
    columns=prediction_columns
)

# Numeric columns — copy directly
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
for col in numeric_cols:
    if col in input_aligned.columns:
        input_aligned.at[0, col] = input_dict[col]

# Categorical columns — set the matching dummy column to 1.
# If the dummy column doesn't exist in X_test.columns, that means this
# value IS the training data's reference category, so leaving it at 0 is correct.
categorical_map = {
    "Gender": gender,
    "family_history_with_overweight": family_history,
    "FAVC": favc,
    "CAEC": caec,
    "SMOKE": smoke,
    "SCC": scc,
    "CALC": calc,
    "MTRANS": mtrans,
}

for col, value in categorical_map.items():
    dummy_col_name = f"{col}_{value}"
    if dummy_col_name in input_aligned.columns:
        input_aligned.at[0, dummy_col_name] = 1.0

raw_prediction = model.predict(
    input_aligned
)[0]

prediction_proba = model.predict_proba(
    input_aligned
)[0]

# Convert XGBoost's integer class ID back to the original obesity label.
if selected_model_name == "XGBoost":
    prediction = XGB_CLASS_NAMES[int(raw_prediction)]
    probability_class_names = XGB_CLASS_NAMES
else:
    prediction = str(raw_prediction)
    probability_class_names = np.asarray(model.classes_).astype(str)

prediction_probability = prediction_proba.max()


proba_df = pd.DataFrame({
    "Class": probability_class_names,
    "Probability": prediction_proba
}).sort_values(
    "Probability",
    ascending=False
)


# ============================================================
# MAIN AREA — PERFORMANCE OVERVIEW
# ============================================================

st.header(
    f"{selected_model_name} — Performance Overview"
)

st.caption(
    "Model performance and prediction results based on "
    "the currently selected model."
)


# ============================================================
# PERFORMANCE METRICS
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

prediction_col1, prediction_col2 = st.columns(
    [1, 2]
)


# ------------------------------------------------------------
# LEFT — PREDICTED CLASS
# ------------------------------------------------------------

with prediction_col1:

    st.markdown("### Predicted Obesity Level")

    st.success(
        f"## {prediction}"
    )

    st.metric(
        "Prediction Probability",
        f"{prediction_probability:.2%}"
    )

    st.caption(
        f"Predicted using the {selected_model_name} model."
    )


# ------------------------------------------------------------
# RIGHT — PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

with prediction_col2:

    st.markdown("### Class Probabilities")

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.barh(
        proba_df["Class"][::-1],
        proba_df["Probability"][::-1],
        color="steelblue"
    )

    ax.set_xlabel(
        "Predicted Probability"
    )

    ax.set_xlim(
        0,
        1
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
    gc.collect()


# ============================================================
# INPUT SUMMARY
# ============================================================

st.markdown("---")

st.subheader("Prediction Input Summary")

input_summary_col1, input_summary_col2, input_summary_col3 = (
    st.columns(3)
)


with input_summary_col1:

    st.metric(
        "Age",
        f"{age} years"
    )

    st.metric(
        "Gender",
        gender
    )

    st.metric(
        "Height",
        f"{height:.2f} m"
    )


with input_summary_col2:

    st.metric(
        "Weight",
        f"{weight:.1f} kg"
    )

    st.metric(
        "BMI",
        f"{bmi:.2f}"
    )

    st.metric(
        "Family History",
        family_history
    )


with input_summary_col3:

    st.metric(
        "Physical Activity",
        f"{faf:.1f}"
    )

    st.metric(
        "Water Consumption",
        f"{ch2o:.1f}"
    )

    st.metric(
        "Technology Usage",
        f"{tue:.1f}"
    )


st.markdown("---")


# ============================================================
# RESULTS NAVIGATION
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
        selected_config["confusion_matrix_image"],
        width="stretch"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

elif section == "Classification Report":

    st.subheader("Classification Report")

    st.caption(
        "Detailed precision, recall, F1-score and support "
        "for each obesity category."
    )

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
    st.caption(f"Feature importance generated for the {selected_model_name} model.")

    st.image(selected_config["feature_importance_image"], width="stretch")

    if hasattr(model, "feature_importances_"):
        if selected_model_name == "XGBoost":
            feature_names = model.get_booster().feature_names
            if not feature_names:
                feature_names = X_test.columns.tolist()
        else:
            feature_names = X_test.columns.tolist()

        importances = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        top15 = importances.head(15)
        feature_df = top15.reset_index().rename(
            columns={"index": "Feature", 0: "Importance"}
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)

    elif "feature_importance_data" in selected_config:
        feature_df = pd.read_csv(selected_config["feature_importance_data"])
        st.dataframe(feature_df, width="stretch", hide_index=True)

    else:
        st.info("Feature importance is not available for this model.")

# ============================================================
# ROC CURVES
# ============================================================

elif section == "ROC Curves":

    st.subheader("ROC Curves")

    st.caption(
        f"One-vs-Rest ROC curves for the "
        f"{selected_model_name} model."
    )

    st.image(
        selected_config["roc_curve_image"],
        width="stretch"
    )

# ============================================================
# MODEL COMPARISON — DASHBOARD
# ============================================================

st.markdown("---")
st.header("Model Comparison")
st.caption(
    "Compare all four trained models using the same held-out test set. "
    "The dashboard highlights the strongest overall model and shows where the models differ."
)

# ------------------------------------------------------------
# Build one numeric comparison dataset
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_comparison_data():
    rows = []

    for model_name, config in MODEL_REGISTRY.items():
        if not config.get("available", False):
            continue

        try:
            loaded_model, _, loaded_y_test, loaded_y_pred, loaded_y_proba = load_artifacts(config)

            loaded_y_test = np.asarray(loaded_y_test).ravel().astype(str)
            loaded_y_pred = np.asarray(loaded_y_pred).ravel().astype(str)

            if model_name == "XGBoost":
                loaded_classes = XGB_CLASS_NAMES
            else:
                loaded_classes = np.asarray(loaded_model.classes_).astype(str)

            loaded_y_test_bin = label_binarize(
                loaded_y_test,
                classes=loaded_classes
            )

            rows.append({
                "Model": model_name,
                "Accuracy": accuracy_score(loaded_y_test, loaded_y_pred),
                "F1 Score": f1_score(loaded_y_test, loaded_y_pred, average="weighted"),
                "Precision": precision_score(loaded_y_test, loaded_y_pred, average="weighted"),
                "Recall": recall_score(loaded_y_test, loaded_y_pred, average="weighted"),
                "ROC-AUC": roc_auc_score(
                    loaded_y_test_bin,
                    loaded_y_proba,
                    average="macro",
                    multi_class="ovr"
                ),
                "Misclassification": (loaded_y_test != loaded_y_pred).mean(),
                "Correct": int((loaded_y_test == loaded_y_pred).sum()),
                "Errors": int((loaded_y_test != loaded_y_pred).sum()),
                "Test Size": len(loaded_y_test),
            })
        except Exception as exc:
            st.warning(f"Could not calculate comparison metrics for {model_name}: {exc}")

    return pd.DataFrame(rows)


comparison_numeric = build_comparison_data()

if not comparison_numeric.empty:

    # --------------------------------------------------------
    # Overall score / ranking
    # --------------------------------------------------------
    score_metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]
    comparison_numeric["Overall Score"] = comparison_numeric[score_metrics].mean(axis=1)
    comparison_numeric = comparison_numeric.sort_values("Overall Score", ascending=False).reset_index(drop=True)
    comparison_numeric["Rank"] = np.arange(1, len(comparison_numeric) + 1)

    winner = comparison_numeric.iloc[0]
    runner_up = comparison_numeric.iloc[1] if len(comparison_numeric) > 1 else None

    # --------------------------------------------------------
    # Polished CSS cards
    # --------------------------------------------------------
    st.markdown("""
    <style>
    .winner-card {
        padding: 22px 26px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,193,7,.16), rgba(255,255,255,.04));
        border: 1px solid rgba(255,193,7,.45);
        margin: 8px 0 20px 0;
    }
    .winner-title { font-size: 14px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .75; }
    .winner-model { font-size: 30px; font-weight: 800; margin: 2px 0 4px 0; }
    .winner-text { font-size: 15px; opacity: .82; }
    .model-card {
        padding: 18px 18px 14px 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.045);
        min-height: 150px;
    }
    .model-card-best {
        border: 2px solid rgba(255,193,7,.65);
        background: linear-gradient(135deg, rgba(255,193,7,.12), rgba(128,128,128,.035));
    }
    .model-name { font-weight: 750; font-size: 16px; margin-bottom: 8px; }
    .model-rank { float: right; opacity: .65; font-size: 13px; }
    .metric-big { font-size: 28px; font-weight: 800; line-height: 1.1; }
    .metric-label { font-size: 12px; opacity: .68; }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Winner banner
    # --------------------------------------------------------
    gap_text = ""
    if runner_up is not None:
        gap_text = (
            f" It leads the second-ranked {runner_up['Model']} by "
            f"{(winner['Overall Score'] - runner_up['Overall Score']):.3f} "
            "on the average of the five performance metrics."
        )

    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-title">🏆 Best Overall Model</div>
            <div class="winner-model">{winner['Model']}</div>
            <div class="winner-text">
                Strongest combined performance across Accuracy, F1, Precision, Recall and ROC-AUC.{gap_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Four model cards
    # --------------------------------------------------------
    card_cols = st.columns(len(comparison_numeric))

    for col, (_, row) in zip(card_cols, comparison_numeric.iterrows()):
        best_class = "model-card-best" if row["Model"] == winner["Model"] else ""
        badge = "🏆 Best" if row["Model"] == winner["Model"] else f"#{int(row['Rank'])}"
        with col:
            st.markdown(
                f"""
                <div class="model-card {best_class}">
                    <div class="model-name">{row['Model']} <span class="model-rank">{badge}</span></div>
                    <div class="metric-big">{row['Accuracy']:.2%}</div>
                    <div class="metric-label">Accuracy</div>
                    <div style="margin-top:10px;">
                        <b>F1</b> {row['F1 Score']:.3f}
                        &nbsp;&nbsp; <b>AUC</b> {row['ROC-AUC']:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Interactive metric selector
    # --------------------------------------------------------
    left, right = st.columns([1, 2])
    with left:
        comparison_metric = st.selectbox(
            "Compare by",
            ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC", "Misclassification"],
            key="comparison_metric"
        )

    chart_df = comparison_numeric.copy()
    chart_df = chart_df.sort_values(comparison_metric, ascending=(comparison_metric == "Misclassification"))

    # --------------------------------------------------------
    # Main comparison chart
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    values = chart_df[comparison_metric].to_numpy()
    labels = chart_df["Model"].to_numpy()
    bars = ax.barh(labels, values)

    for bar, value in zip(bars, values):
        label = f"{value:.2%}" if comparison_metric != "Misclassification" else f"{value:.2%}"
        ax.text(
            bar.get_width() + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    if comparison_metric == "Misclassification":
        ax.set_xlim(0, max(values) * 1.22 if max(values) else 1)
        ax.set_xlabel("Error Rate")
    else:
        ax.set_xlim(0, min(1.0, max(values) * 1.12))
        ax.set_xlabel("Score")

    ax.set_title(f"{comparison_metric} by Model", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # --------------------------------------------------------
    # Full comparison table
    # --------------------------------------------------------
    st.subheader("Performance Summary")
    st.caption("Higher is better for Accuracy, F1, Precision, Recall and ROC-AUC. Lower is better for Misclassification.")

    display_df = comparison_numeric[[
        "Rank", "Model", "Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC", "Misclassification", "Errors"
    ]].copy()

    display_df.columns = [
        "Rank", "Model", "Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC", "Error Rate", "Errors"
    ]

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2%}",
            "F1 Score": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Error Rate": "{:.2%}",
        }).background_gradient(
            subset=["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"],
            cmap="Blues"
        ).background_gradient(
            subset=["Error Rate"],
            cmap="Reds"
        ),
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # Error comparison + interpretation
    # --------------------------------------------------------
    st.subheader("Classification Errors")

    error_cols = st.columns(len(comparison_numeric))
    for col, (_, row) in zip(error_cols, comparison_numeric.iterrows()):
        with col:
            st.metric(
                row["Model"],
                f"{row['Errors']} errors",
                f"{row['Misclassification']:.2%} error rate",
                delta_color="inverse"
            )

    st.markdown("### What the comparison tells us")

    st.info(
        f"**{winner['Model']} is the strongest overall model** based on the average of the five "
        f"main performance metrics. It achieves **{winner['Accuracy']:.2%} accuracy**, "
        f"a weighted F1-score of **{winner['F1 Score']:.4f}**, and ROC-AUC of **{winner['ROC-AUC']:.4f}**. "
        f"It made **{int(winner['Errors'])} classification errors out of {int(winner['Test Size'])} test cases**. "
        "For this dataset, the model with the strongest overall test performance is therefore the most suitable "
        "candidate for the final prediction system."
    )

    # --------------------------------------------------------
    # Optional: detailed pairwise view
    # --------------------------------------------------------
    with st.expander("View detailed metric differences"):
        detail_metric = st.selectbox(
            "Metric",
            ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"],
            key="detail_metric"
        )
        detail = comparison_numeric[["Model", detail_metric]].copy()
        detail["Difference from best"] = detail[detail_metric] - winner[detail_metric]
        detail["Difference from best"] = detail["Difference from best"].map(lambda x: f"{x:+.4f}")
        st.dataframe(detail, width="stretch", hide_index=True)

else:
    st.warning("No model evaluation artifacts are currently available for comparison.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Obesity Levels dataset — UCI Machine Learning Repository"
)
