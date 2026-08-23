"""
Obesity Level Classification — Random Forest Streamlit Dashboard
================================================================

Run:
    streamlit run app.py

IMPORTANT:
- Uses the exact trained Random Forest model.
- Uses the exact 23 training features.
- Does NOT modify prediction probabilities.
- Does NOT cache the model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import hashlib
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Obesity Level Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(
    "Obesity Level Classification — Random Forest"
)

st.caption(
    "CRISP-DM Project | Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/random_forest_model.pkl"
FEATURE_PATH = "data/model_features.pkl"

X_TEST_PATH = "data/X_test_encoded.pkl"
Y_TEST_PATH = "data/y_test_flat.pkl"
Y_PRED_PATH = "data/rf_y_pred.pkl"
Y_PROBA_PATH = "data/rf_y_pred_proba.pkl"

CONFUSION_IMAGE = "images/rf_confusion_matrix.png"
FEATURE_IMPORTANCE_IMAGE = "images/rf_feature_importance.png"
ROC_IMAGE = "images/rf_roc_curves.png"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    FEATURE_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    Y_PRED_PATH,
    Y_PROBA_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    st.error("Required model files are missing:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(path):

    with open(path, "rb") as f:

        return hashlib.md5(
            f.read()
        ).hexdigest()


# ============================================================
# LOAD MODEL AND DATA
#
# NO STREAMLIT CACHE
# ============================================================

def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    model_features = joblib.load(
        FEATURE_PATH
    )

    X_test = joblib.load(
        X_TEST_PATH
    )

    y_test = joblib.load(
        Y_TEST_PATH
    )

    y_pred = joblib.load(
        Y_PRED_PATH
    )

    y_proba = joblib.load(
        Y_PROBA_PATH
    )

    return (
        model,
        model_features,
        X_test,
        y_test,
        y_pred,
        y_proba
    )


(
    model,
    model_features,
    X_test,
    y_test,
    y_pred,
    y_proba
) = load_model()


# ============================================================
# MODEL HASH
# ============================================================

model_hash = get_file_hash(
    MODEL_PATH
)


# ============================================================
# MODEL VALIDATION
# ============================================================

if not hasattr(
    model,
    "feature_importances_"
):

    st.error(
        "The loaded model is not a Random Forest model."
    )

    st.stop()


# ------------------------------------------------------------
# Check number of features
# ------------------------------------------------------------

if model.n_features_in_ != len(model_features):

    st.error(
        f"""
        Model-feature mismatch.

        Random Forest expects:
        {model.n_features_in_} features

        model_features.pkl contains:
        {len(model_features)} features
        """
    )

    st.stop()


# ------------------------------------------------------------
# Check test dataset features
# ------------------------------------------------------------

if X_test.shape[1] != model.n_features_in_:

    st.error(
        f"""
        X_test feature mismatch.

        Model expects:
        {model.n_features_in_}

        X_test contains:
        {X_test.shape[1]}
        """
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Controls"
)

st.sidebar.subheader(
    "Prediction Inputs"
)


# ============================================================
# DEMOGRAPHIC / PHYSICAL
# ============================================================

st.sidebar.markdown(
    "**Demographic & Physical Information**"
)


gender = st.sidebar.selectbox(
    "Gender",
    [
        "Female",
        "Male"
    ],
    index=0
)


age = st.sidebar.number_input(
    "Age (years)",
    min_value=18,
    max_value=100,
    value=25,
    step=1
)


# IMPORTANT:
# Dataset training range is approximately:
# Height: 1.45–1.98 m

height = st.sidebar.number_input(
    "Height (m)",
    min_value=1.45,
    max_value=1.98,
    value=1.60,
    step=0.01
)


# IMPORTANT:
# Dataset training range is approximately:
# Weight: 39–173 kg

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=39.0,
    max_value=173.0,
    value=60.0,
    step=0.5
)


family_history = st.sidebar.selectbox(
    "Family history with overweight?",
    [
        "yes",
        "no"
    ],
    index=1
)


# ============================================================
# BMI
# ============================================================

bmi = (
    float(weight)
    /
    (float(height) ** 2)
)


st.sidebar.metric(
    "Calculated BMI",
    f"{bmi:.2f}"
)


# ============================================================
# EATING HABITS
# ============================================================

st.sidebar.markdown(
    "**Eating Habits**"
)


favc = st.sidebar.selectbox(
    "Frequent high-calorie food (FAVC)?",
    [
        "yes",
        "no"
    ],
    index=0
)


fcvc = st.sidebar.slider(
    "Vegetable consumption frequency (FCVC)",
    1.0,
    3.0,
    2.0,
    step=0.1
)


ncp = st.sidebar.slider(
    "Number of main meals (NCP)",
    1.0,
    4.0,
    3.0,
    step=0.1
)


caec = st.sidebar.selectbox(
    "Eating between meals (CAEC)",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
    index=1
)


ch2o = st.sidebar.slider(
    "Daily water consumption (CH2O)",
    1.0,
    3.0,
    2.0,
    step=0.1
)


scc = st.sidebar.selectbox(
    "Monitors calorie consumption (SCC)?",
    [
        "no",
        "yes"
    ],
    index=0
)


calc = st.sidebar.selectbox(
    "Alcohol consumption (CALC)",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
    index=1
)


# ============================================================
# LIFESTYLE
# ============================================================

st.sidebar.markdown(
    "**Lifestyle**"
)


smoke = st.sidebar.selectbox(
    "Smoker?",
    [
        "no",
        "yes"
    ],
    index=0
)


faf = st.sidebar.slider(
    "Physical activity frequency (FAF)",
    0.0,
    3.0,
    1.0,
    step=0.1
)


tue = st.sidebar.slider(
    "Time using technology devices (TUE)",
    0.0,
    2.0,
    1.0,
    step=0.1
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
    index=3
)


# ============================================================
# BUILD INPUT DATA
# ============================================================

input_data = {

    "Age":
        float(age),

    "Height":
        float(height),

    "Weight":
        float(weight),

    "FCVC":
        float(fcvc),

    "NCP":
        float(ncp),

    "CH2O":
        float(ch2o),

    "FAF":
        float(faf),

    "TUE":
        float(tue),

    "Gender_Male":
        1 if gender == "Male" else 0,

    "family_history_with_overweight_yes":
        1 if family_history == "yes" else 0,

    "FAVC_yes":
        1 if favc == "yes" else 0,

    "CAEC_Frequently":
        1 if caec == "Frequently" else 0,

    "CAEC_Sometimes":
        1 if caec == "Sometimes" else 0,

    "CAEC_no":
        1 if caec == "no" else 0,

    "SMOKE_yes":
        1 if smoke == "yes" else 0,

    "SCC_yes":
        1 if scc == "yes" else 0,

    "CALC_Frequently":
        1 if calc == "Frequently" else 0,

    "CALC_Sometimes":
        1 if calc == "Sometimes" else 0,

    "CALC_no":
        1 if calc == "no" else 0,

    "MTRANS_Bike":
        1 if mtrans == "Bike" else 0,

    "MTRANS_Motorbike":
        1 if mtrans == "Motorbike" else 0,

    "MTRANS_Public_Transportation":
        1 if mtrans == "Public_Transportation" else 0,

    "MTRANS_Walking":
        1 if mtrans == "Walking" else 0
}


input_df = pd.DataFrame(
    [input_data]
)


# ============================================================
# FORCE EXACT TRAINING FEATURE ORDER
# ============================================================

input_df = input_df.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================================
# FORCE NUMERIC DATA TYPE
# ============================================================

input_df = input_df.astype(
    np.float64
)


# ============================================================
# VERIFY FEATURE NAMES
# ============================================================

if list(input_df.columns) != list(model_features):

    st.error(
        "Feature order does not match model_features.pkl."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_df
)[0]


prediction_proba = model.predict_proba(
    input_df
)[0]


prediction_probability = float(
    np.max(prediction_proba)
)


# ============================================================
# PROBABILITY DATAFRAME
# ============================================================

proba_df = pd.DataFrame({

    "Class":
        model.classes_,

    "Probability":
        prediction_proba

}).sort_values(
    "Probability",
    ascending=False
)


# ============================================================
# PERFORMANCE METRICS
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


y_test_binarized = label_binarize(
    y_test,
    classes=model.classes_
)


roc_auc_macro = roc_auc_score(
    y_test_binarized,
    y_proba,
    average="macro",
    multi_class="ovr"
)


# ============================================================
# HEADER
# ============================================================

st.header(
    "Random Forest — Performance Overview"
)


# ============================================================
# PERFORMANCE CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)


c2.metric(
    "F1 (Weighted)",
    f"{f1_weighted:.4f}"
)


c3.metric(
    "Precision",
    f"{precision_w:.4f}"
)


c4.metric(
    "Recall",
    f"{recall_w:.4f}"
)


c5.metric(
    "ROC-AUC",
    f"{roc_auc_macro:.4f}"
)


# ============================================================
# PREDICTION RESULT
# ============================================================

st.markdown("---")

st.subheader(
    "Prediction Result"
)


p1, p2 = st.columns(
    [1, 2]
)


with p1:

    st.markdown(
        "### Predicted Obesity Level"
    )

    st.success(
        f"## {prediction}"
    )

    st.metric(
        "Model Confidence",
        f"{prediction_probability:.2%}"
    )

    # --------------------------------------------------------
    # Confidence explanation
    # --------------------------------------------------------

    if prediction_probability >= 0.80:

        st.success(
            "High model confidence"
        )

    elif prediction_probability >= 0.60:

        st.info(
            "Moderate model confidence"
        )

    else:

        st.warning(
            "Lower model confidence — "
            "the model finds the input less clearly "
            "associated with one obesity category."
        )

    st.caption(
        "Model confidence is the Random Forest's highest "
        "predicted class probability. It is not a guarantee "
        "that the prediction is correct."
    )


with p2:

    st.markdown(
        "### Class Probabilities"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.barh(
        proba_df["Class"][::-1],
        proba_df["Probability"][::-1]
    )

    ax.set_xlabel(
        "Probability"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_title(
        "Random Forest Class Probabilities"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


# ============================================================
# INPUT SUMMARY
# ============================================================

st.markdown("---")

st.subheader(
    "Prediction Input Summary"
)


c1, c2, c3 = st.columns(3)


with c1:

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


with c2:

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


with c3:

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


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.header(
    "Model Comparison"
)


comparison_df = pd.DataFrame({

    "Model": [
        "Random Forest",
        "K-Nearest Neighbours (KNN)",
        "Logistic Regression",
        "Gradient Boosting"
    ],

    "Accuracy": [
        f"{accuracy:.2%}",
        "Not Available",
        "Not Available",
        "Not Available"
    ],

    "F1 (Weighted)": [
        f"{f1_weighted:.4f}",
        "Not Available",
        "Not Available",
        "Not Available"
    ],

    "Precision (Weighted)": [
        f"{precision_w:.4f}",
        "Not Available",
        "Not Available",
        "Not Available"
    ],

    "Recall (Weighted)": [
        f"{recall_w:.4f}",
        "Not Available",
        "Not Available",
        "Not Available"
    ],

    "ROC-AUC (Macro)": [
        f"{roc_auc_macro:.4f}",
        "Not Available",
        "Not Available",
        "Not Available"
    ]
})


st.dataframe(
    comparison_df,
    hide_index=True,
    width="stretch"
)


# ============================================================
# RESULTS
# ============================================================

st.markdown("---")

section = st.radio(
    "View Results",
    [
        "Confusion Matrix",
        "Classification Report",
        "Feature Importance",
        "ROC Curves"
    ],
    horizontal=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

if section == "Confusion Matrix":

    st.subheader(
        "Confusion Matrix"
    )

    if os.path.exists(CONFUSION_IMAGE):

        st.image(
            CONFUSION_IMAGE,
            width="stretch"
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

elif section == "Classification Report":

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
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

    st.subheader(
        "Feature Importance"
    )

    if os.path.exists(FEATURE_IMPORTANCE_IMAGE):

        st.image(
            FEATURE_IMPORTANCE_IMAGE,
            width="stretch"
        )

    else:

        st.warning(
            "Feature importance image not found."
        )


    importances = pd.Series(
        model.feature_importances_,
        index=model_features
    ).sort_values(
        ascending=False
    )


    feature_df = (
        importances
        .head(15)
        .reset_index()
    )


    feature_df.columns = [
        "Feature",
        "Importance"
    ]


    st.dataframe(
        feature_df,
        hide_index=True,
        width="stretch"
    )


# ============================================================
# ROC CURVES
# ============================================================

elif section == "ROC Curves":

    st.subheader(
        "ROC Curves"
    )

    if os.path.exists(ROC_IMAGE):

        st.image(
            ROC_IMAGE,
            width="stretch"
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

with st.expander("Model Information"):

    st.write(
        "Model type:",
        type(model).__name__
    )

    st.write(
        "Number of features:",
        model.n_features_in_
    )

    st.write(
        "Number of trees:",
        model.n_estimators
    )

    st.write(
        "Classes:",
        list(model.classes_)
    )

    st.write(
        "Model MD5:",
        model_hash
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Obesity Levels dataset — "
    "UCI Machine Learning Repository"
)
