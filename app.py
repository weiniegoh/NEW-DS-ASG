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
# MODEL COMPARISON — HIGH-MARK ANALYTICS DASHBOARD
# ============================================================

st.markdown("---")
st.header("Model Comparison")
st.caption(
    "Analytical comparison of Logistic Regression, KNN, Random Forest and "
    "XGBoost using the same 627-sample held-out test set."
)


# ------------------------------------------------------------
# Notebook cross-validation results
# ------------------------------------------------------------
# These values come directly from the project's notebook.
# Test-set metrics and misclassifications are still calculated
# dynamically from the saved prediction artifacts.
CV_RESULTS = {
    "Logistic Regression": {
        "CV Accuracy": 0.8568,
        "CV Accuracy Std": 0.0214,
        "CV F1": 0.8542,
        "CV F1 Std": 0.0227,
    },
    "Random Forest": {
        "CV Accuracy": 0.9253,
        "CV Accuracy Std": 0.0416 / 2,
        "CV F1": 0.9264,
        "CV F1 Std": 0.0421 / 2,
    },
    "K-Nearest Neighbours (KNN)": {
        "CV Accuracy": 0.8822,
        "CV Accuracy Std": 0.0148,
        "CV F1": 0.8779,
        "CV F1 Std": 0.0165,
    },
    "XGBoost": {
        "CV Accuracy": 0.9651,
        "CV Accuracy Std": 0.0105,
        "CV F1": 0.9649,
        "CV F1 Std": 0.0105,
    },
}


# ------------------------------------------------------------
# Load all model artifacts once
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_model_comparison():
    rows = []
    artifacts = {}

    for model_name, config in MODEL_REGISTRY.items():
        if not config.get("available", False):
            continue

        try:
            loaded_model, loaded_X_test, loaded_y_test, loaded_y_pred, loaded_y_proba = (
                load_artifacts(config)
            )

            y_true = np.asarray(loaded_y_test).ravel().astype(str)
            y_pred = np.asarray(loaded_y_pred).ravel().astype(str)

            # Ensure probability output is numeric and 2-D.
            y_proba = np.asarray(loaded_y_proba)

            if model_name == "XGBoost":
                classes = XGB_CLASS_NAMES
            else:
                classes = np.asarray(loaded_model.classes_).astype(str)

            y_true_bin = label_binarize(y_true, classes=classes)

            # Some artifacts can contain one extra dimension.
            if y_proba.ndim > 2:
                y_proba = np.squeeze(y_proba)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            weighted_f1 = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            macro_f1 = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )

            try:
                roc_auc = roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average="weighted",
                    multi_class="ovr",
                )
            except ValueError:
                roc_auc = np.nan

            errors = int(np.sum(y_true != y_pred))
            error_rate = errors / len(y_true)

            cv = CV_RESULTS.get(model_name, {})

            rows.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": weighted_f1,
                "Macro F1": macro_f1,
                "ROC-AUC": roc_auc,
                "Errors": errors,
                "Error Rate": error_rate,
                "Test Size": len(y_true),
                "CV Accuracy": cv.get("CV Accuracy", np.nan),
                "CV Accuracy Std": cv.get("CV Accuracy Std", np.nan),
                "CV F1": cv.get("CV F1", np.nan),
                "CV F1 Std": cv.get("CV F1 Std", np.nan),
            })

            artifacts[model_name] = {
                "model": loaded_model,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "classes": classes,
            }

        except Exception as exc:
            st.warning(
                f"Could not load comparison results for {model_name}: {exc}"
            )

    return pd.DataFrame(rows), artifacts


comparison_df, comparison_artifacts = build_model_comparison()

if comparison_df.empty:
    st.error("No model evaluation artifacts are available.")
else:

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------
    score_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ]

    comparison_df["Overall Score"] = comparison_df[score_metrics].mean(
        axis=1, skipna=True
    )

    comparison_df = (
        comparison_df
        .sort_values("Overall Score", ascending=False)
        .reset_index(drop=True)
    )

    comparison_df["Rank"] = np.arange(1, len(comparison_df) + 1)

    winner = comparison_df.iloc[0]

    # --------------------------------------------------------
    # CSS
    # --------------------------------------------------------
    st.markdown(
        """
        <style>
        .comparison-winner {
            padding: 24px;
            border-radius: 16px;
            border: 1px solid rgba(46, 125, 50, .35);
            background: rgba(46, 125, 50, .08);
            margin-bottom: 18px;
        }

        .comparison-winner h2 {
            margin: 0 0 6px 0;
        }

        .comparison-card {
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.25);
            min-height: 180px;
        }

        .comparison-card-best {
            border: 2px solid rgba(46,125,50,.65);
            background: rgba(46,125,50,.07);
        }

        .small-label {
            font-size: 12px;
            opacity: .70;
        }

        .large-number {
            font-size: 28px;
            font-weight: 800;
        }

        .analysis-box {
            padding: 18px;
            border-radius: 12px;
            background: rgba(128,128,128,.06);
            border: 1px solid rgba(128,128,128,.20);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 1. WINNER / MODEL SELECTION
    # --------------------------------------------------------
    st.markdown(
        f"""
        <div class="comparison-winner">
            <h2>🏆 Recommended Final Model: {winner["Model"]}</h2>
            <div>
                The model has the strongest overall test performance across
                Accuracy, Precision, Recall, weighted F1-score and ROC-AUC.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    winner_cols = st.columns(5)

    winner_metrics = [
        ("Accuracy", winner["Accuracy"], "{:.2%}"),
        ("Weighted F1", winner["F1 Score"], "{:.4f}"),
        ("Precision", winner["Precision"], "{:.4f}"),
        ("Recall", winner["Recall"], "{:.4f}"),
        ("ROC-AUC", winner["ROC-AUC"], "{:.4f}"),
    ]

    for col, (label, value, fmt) in zip(winner_cols, winner_metrics):
        with col:
            st.metric(label, fmt.format(value))

    # --------------------------------------------------------
    # Four model cards
    # --------------------------------------------------------
    st.markdown("### Model Scorecards")

    card_cols = st.columns(len(comparison_df))

    for col, (_, row) in zip(card_cols, comparison_df.iterrows()):
        best_class = (
            "comparison-card-best"
            if row["Model"] == winner["Model"]
            else ""
        )

        with col:
            st.markdown(
                f"""
                <div class="comparison-card {best_class}">
                    <div>
                        <b>#{int(row["Rank"])} {row["Model"]}</b>
                    </div>
                    <br>
                    <div class="small-label">Accuracy</div>
                    <div class="large-number">{row["Accuracy"]:.2%}</div>
                    <br>
                    <div>
                        <b>F1:</b> {row["F1 Score"]:.4f}<br>
                        <b>ROC-AUC:</b> {row["ROC-AUC"]:.4f}<br>
                        <b>Errors:</b> {int(row["Errors"])} / {int(row["Test Size"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # --------------------------------------------------------
    # 2. INTERACTIVE METRIC COMPARISON
    # --------------------------------------------------------
    st.markdown("### 1. Performance Comparison")

    metric = st.selectbox(
        "Select a metric to compare",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Error Rate",
        ],
        key="comparison_metric_high_mark",
    )

    chart_df = comparison_df[["Model", metric]].copy()

    # For error rate, lower is better.
    ascending = metric == "Error Rate"
    chart_df = chart_df.sort_values(metric, ascending=ascending)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(
        chart_df["Model"],
        chart_df[metric],
    )

    max_value = float(chart_df[metric].max())

    for bar, value in zip(bars, chart_df[metric]):
        ax.text(
            bar.get_width() + max(max_value * 0.015, 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2%}",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel(
        "Error Rate" if metric == "Error Rate" else "Score"
    )
    ax.set_title(
        f"{metric} Comparison Across Models",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.20)
    ax.spines[["top", "right", "left"]].set_visible(False)

    upper = max_value * 1.18 if max_value > 0 else 1
    ax.set_xlim(0, upper)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # --------------------------------------------------------
    # Performance summary table
    # --------------------------------------------------------
    st.markdown("#### Complete Performance Summary")

    summary_display = comparison_df[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Errors",
            "Error Rate",
        ]
    ].copy()

    summary_display.columns = [
        "Rank",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1",
        "ROC-AUC",
        "Errors",
        "Error Rate",
    ]

    st.dataframe(
        summary_display.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Error Rate": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 3. CROSS-VALIDATION STABILITY
    # --------------------------------------------------------
    st.markdown("### 2. Cross-Validation Stability")

    st.caption(
        "Five-fold cross-validation on the training data shows whether "
        "performance is consistent across different training folds."
    )

    cv_display = comparison_df[
        [
            "Model",
            "CV Accuracy",
            "CV Accuracy Std",
            "CV F1",
            "CV F1 Std",
            "Accuracy",
            "F1 Score",
        ]
    ].copy()

    cv_display.columns = [
        "Model",
        "CV Accuracy",
        "CV Accuracy SD",
        "CV Weighted F1",
        "CV F1 SD",
        "Test Accuracy",
        "Test Weighted F1",
    ]

    st.dataframe(
        cv_display.style.format({
            "CV Accuracy": "{:.2%}",
            "CV Accuracy SD": "{:.4f}",
            "CV Weighted F1": "{:.4f}",
            "CV F1 SD": "{:.4f}",
            "Test Accuracy": "{:.2%}",
            "Test Weighted F1": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    cv_fig, cv_ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(comparison_df))
    width = 0.35

    cv_ax.bar(
        x - width / 2,
        comparison_df["CV Accuracy"],
        width,
        label="5-Fold CV Accuracy",
    )

    cv_ax.bar(
        x + width / 2,
        comparison_df["Accuracy"],
        width,
        label="Test Accuracy",
    )

    cv_ax.set_xticks(x)
    cv_ax.set_xticklabels(comparison_df["Model"], rotation=15)
    cv_ax.set_ylabel("Accuracy")
    cv_ax.set_ylim(
        0,
        min(1.0, max(
            comparison_df["CV Accuracy"].max(),
            comparison_df["Accuracy"].max(),
        ) * 1.12),
    )
    cv_ax.set_title(
        "Cross-Validation vs Held-Out Test Accuracy",
        fontweight="bold",
    )
    cv_ax.legend()
    cv_ax.grid(axis="y", alpha=0.20)

    plt.tight_layout()
    st.pyplot(cv_fig, width="stretch")
    plt.close(cv_fig)

    # Stability interpretation
    best_cv = comparison_df.sort_values(
        "CV Accuracy",
        ascending=False,
    ).iloc[0]

    st.info(
        f"**Stability insight:** {best_cv['Model']} has the highest "
        f"five-fold CV accuracy at **{best_cv['CV Accuracy']:.2%}**. "
        f"{winner['Model']} also achieves a test accuracy of "
        f"**{winner['Accuracy']:.2%}**, indicating strong predictive "
        f"performance on the held-out test set."
    )

    # --------------------------------------------------------
    # 4. CLASS-LEVEL PERFORMANCE
    # --------------------------------------------------------
    st.markdown("### 3. Class-Level Performance")

    class_model = st.selectbox(
        "Select a model for class-level analysis",
        list(comparison_artifacts.keys()),
        key="class_level_model",
    )

    class_artifact = comparison_artifacts[class_model]

    report_dict = classification_report(
        class_artifact["y_true"],
        class_artifact["y_pred"],
        labels=class_artifact["classes"],
        output_dict=True,
        zero_division=0,
    )

    class_rows = []

    for class_name in class_artifact["classes"]:
        if class_name in report_dict:
            class_rows.append({
                "Class": class_name,
                "Precision": report_dict[class_name]["precision"],
                "Recall": report_dict[class_name]["recall"],
                "F1 Score": report_dict[class_name]["f1-score"],
                "Support": int(report_dict[class_name]["support"]),
            })

    class_df = pd.DataFrame(class_rows)

    if not class_df.empty:

        st.dataframe(
            class_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        class_fig, class_ax = plt.subplots(figsize=(10, 5))

        bars = class_ax.barh(
            class_df["Class"],
            class_df["F1 Score"],
        )

        for bar, value in zip(bars, class_df["F1 Score"]):
            class_ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontweight="bold",
            )

        class_ax.set_xlim(
            0,
            min(1.0, max(class_df["F1 Score"]) * 1.12),
        )
        class_ax.set_xlabel("F1 Score")
        class_ax.set_title(
            f"Per-Class F1 Score — {class_model}",
            fontweight="bold",
        )
        class_ax.grid(axis="x", alpha=0.20)
        class_ax.spines[["top", "right", "left"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(class_fig, width="stretch")
        plt.close(class_fig)

        weakest = class_df.loc[class_df["F1 Score"].idxmin()]
        strongest = class_df.loc[class_df["F1 Score"].idxmax()]

        st.info(
            f"**Class-level insight:** The strongest class for "
            f"{class_model} is **{strongest['Class']}** "
            f"(F1 = {strongest['F1 Score']:.3f}), while the weakest is "
            f"**{weakest['Class']}** (F1 = {weakest['F1 Score']:.3f}). "
            "This highlights which obesity categories are easier or harder "
            "for the selected model to distinguish."
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    st.markdown("#### Confusion Matrix")

    cm = pd.crosstab(
        pd.Series(
            class_artifact["y_true"],
            name="Actual",
        ),
        pd.Series(
            class_artifact["y_pred"],
            name="Predicted",
        ),
    )

    cm = cm.reindex(
        index=class_artifact["classes"],
        columns=class_artifact["classes"],
        fill_value=0,
    )

    cm_fig, cm_ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=cm_ax,
    )

    cm_ax.set_title(
        f"Confusion Matrix — {class_model}",
        fontweight="bold",
    )
    cm_ax.set_xlabel("Predicted")
    cm_ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(cm_fig, width="stretch")
    plt.close(cm_fig)

    # --------------------------------------------------------
    # 5. MISCLASSIFICATION ANALYSIS
    # --------------------------------------------------------
    st.markdown("### 4. Misclassification Analysis")

    mis_rows = []

    for model_name, artifact in comparison_artifacts.items():

        y_true = artifact["y_true"]
        y_pred = artifact["y_pred"]

        incorrect = y_true != y_pred
        errors = int(incorrect.sum())
        total = len(y_true)

        mis_rows.append({
            "Model": model_name,
            "Misclassified": errors,
            "Total": total,
            "Error Rate": errors / total,
        })

    mis_df = pd.DataFrame(mis_rows).sort_values(
        "Error Rate"
    ).reset_index(drop=True)

    mis_df["Rank"] = np.arange(1, len(mis_df) + 1)

    mis_cols = st.columns(len(mis_df))

    for col, (_, row) in zip(mis_cols, mis_df.iterrows()):
        with col:
            st.metric(
                f"#{int(row['Rank'])} {row['Model']}",
                f"{int(row['Misclassified'])} errors",
                f"{row['Error Rate']:.2%}",
            )

    mis_fig, mis_ax = plt.subplots(figsize=(10, 5))

    bars = mis_ax.barh(
        mis_df["Model"],
        mis_df["Error Rate"],
    )

    for bar, value in zip(bars, mis_df["Error Rate"]):
        mis_ax.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2%}",
            va="center",
            fontweight="bold",
        )

    mis_ax.set_xlabel("Misclassification Rate")
    mis_ax.set_title(
        "Misclassification Rate — Lower is Better",
        fontweight="bold",
    )
    mis_ax.grid(axis="x", alpha=0.20)
    mis_ax.spines[["top", "right", "left"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(mis_fig, width="stretch")
    plt.close(mis_fig)

    selected_mis_model = st.selectbox(
        "Inspect the most common errors",
        list(comparison_artifacts.keys()),
        key="selected_misclassification_model",
    )

    selected_artifact = comparison_artifacts[selected_mis_model]

    error_df = pd.DataFrame({
        "Actual": selected_artifact["y_true"],
        "Predicted": selected_artifact["y_pred"],
    })

    error_df = error_df[
        error_df["Actual"] != error_df["Predicted"]
    ]

    if not error_df.empty:

        top_errors = (
            error_df
            .groupby(["Actual", "Predicted"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(5)
        )

        top_errors["Misclassification"] = (
            top_errors["Actual"].astype(str)
            + " → "
            + top_errors["Predicted"].astype(str)
        )

        top_errors = top_errors[
            ["Misclassification", "Count"]
        ]

        st.dataframe(
            top_errors,
            use_container_width=True,
            hide_index=True,
        )

        total_errors = len(error_df)
        total_cases = len(selected_artifact["y_true"])

        st.caption(
            f"{selected_mis_model} made **{total_errors} errors out of "
            f"{total_cases} test cases ({total_errors / total_cases:.2%})**."
        )

    # --------------------------------------------------------
    # Analytical interpretation
    # --------------------------------------------------------
    st.markdown("#### What the misclassifications tell us")

    xgb_errors = None
    rf_errors = None
    knn_errors = None
    lr_errors = None

    for _, row in mis_df.iterrows():
        if row["Model"] == "XGBoost":
            xgb_errors = int(row["Misclassified"])
        elif row["Model"] == "Random Forest":
            rf_errors = int(row["Misclassified"])
        elif row["Model"] == "K-Nearest Neighbours (KNN)":
            knn_errors = int(row["Misclassified"])
        elif row["Model"] == "Logistic Regression":
            lr_errors = int(row["Misclassified"])

    if xgb_errors is not None and knn_errors is not None:
        st.info(
            f"**XGBoost makes {knn_errors - xgb_errors} fewer errors than KNN** "
            f"({xgb_errors} vs {knn_errors}) on the same 627 test cases. "
            f"Random Forest makes {rf_errors} errors and Logistic Regression "
            f"makes {lr_errors} errors. This provides stronger evidence for "
            "the final model choice than accuracy alone."
        )

    # --------------------------------------------------------
    # 6. MODEL STRENGTHS AND WEAKNESSES
    # --------------------------------------------------------
    st.markdown("### 5. Model Strengths and Weaknesses")

    strengths = {
        "Logistic Regression": (
            "Strong baseline, simple to interpret and computationally efficient.",
            "Lower predictive performance than the tree-based models and KNN."
        ),
        "K-Nearest Neighbours (KNN)": (
            "Simple distance-based approach that can capture local patterns.",
            "Highest error rate among the four models and weaker class separation."
        ),
        "Random Forest": (
            "Captures nonlinear relationships and feature interactions while "
            "providing useful feature-importance information.",
            "More complex than Logistic Regression and still produces more errors "
            "than XGBoost."
        ),
        "XGBoost": (
            "Strongest overall predictive performance, lowest error rate and "
            "excellent ROC-AUC.",
            "More complex to tune and less directly interpretable than Logistic Regression."
        ),
    }

    strength_df = pd.DataFrame([
        {
            "Model": name,
            "Strength": strengths[name][0],
            "Limitation": strengths[name][1],
        }
        for name in comparison_df["Model"]
        if name in strengths
    ])

    st.dataframe(
        strength_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Final model selection
    # --------------------------------------------------------
    st.markdown("### Final Model Selection")

    st.success(
        f"""
        **Selected model: {winner["Model"]}**

        The model is selected because it provides the strongest overall
        combination of Accuracy, Precision, Recall, weighted F1-score and
        ROC-AUC. It also produces the lowest number of classification errors
        on the held-out test set.

        This makes **{winner["Model"]}** the most suitable candidate for the
        final obesity-level prediction system for this dataset.
        """
    )

    # --------------------------------------------------------
    # Limitations and future improvements
    # --------------------------------------------------------
    with st.expander("Limitations and Future Improvements"):

        st.markdown(
            """
            **Limitations**

            - Evaluation is based on the available dataset and a single
              held-out test split of 627 observations.
            - Similar neighbouring obesity categories can overlap in feature
              space, which can lead to confusion between adjacent classes.
            - High predictive performance on this dataset does not guarantee
              identical performance on a different population or external dataset.
            - Feature importance should be interpreted as model association,
              not as proof of causal relationships.

            **Future Improvements**

            - Validate the selected model on an independent external dataset.
            - Investigate the weakest-performing classes in greater detail.
            - Perform additional hyperparameter optimisation where appropriate.
            - Apply explainable-AI methods such as SHAP to improve model
              interpretability.
            - Monitor model performance after deployment.
            """
        )

    # --------------------------------------------------------
    # Detailed metric differences
    # --------------------------------------------------------
    with st.expander("View Detailed Metric Differences"):

        detail_metric = st.selectbox(
            "Select metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ],
            key="detail_metric_high_mark",
        )

        detail_df = comparison_df[
            ["Rank", "Model", detail_metric]
        ].copy()

        best_value = comparison_df.iloc[0][detail_metric]

        detail_df["Difference from Best"] = (
            detail_df[detail_metric] - best_value
        )

        st.dataframe(
            detail_df.style.format({
                detail_metric: "{:.4f}",
                "Difference from Best": "{:+.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Obesity Levels dataset — UCI Machine Learning Repository")
