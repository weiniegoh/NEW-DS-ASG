"""
Obesity Level Classification — Model Comparison Dashboard
============================================================
Run with:

    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import gc

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
    layout="wide"
)

st.title(
    "Obesity Level Classification — Model Comparison"
)

st.caption(
    "CRISP-DM Project | Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "Random Forest": {

        "available": True,

        "model_path":
            "models/random_forest_model.pkl",

        "model_features_path":
            "data/model_features.pkl",

        "X_test_path":
            "data/X_test_encoded.pkl",

        "y_test_path":
            "data/y_test_flat.pkl",

        "y_pred_path":
            "data/rf_y_pred.pkl",

        "y_proba_path":
            "data/rf_y_pred_proba.pkl",

        "confusion_matrix_image":
            "images/rf_confusion_matrix.png",

        "feature_importance_image":
            "images/rf_feature_importance.png",

        "roc_curve_image":
            "images/rf_roc_curves.png",
    },

    "K-Nearest Neighbours (KNN)": {
        "available": False,
    },

    "Logistic Regression": {
        "available": False,
    },

    "Gradient Boosting": {
        "available": False,
    },
}


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts(config):

    model = joblib.load(
        config["model_path"]
    )

    model_features = joblib.load(
        config["model_features_path"]
    )

    X_test = joblib.load(
        config["X_test_path"]
    )

    y_test = joblib.load(
        config["y_test_path"]
    )

    y_pred = joblib.load(
        config["y_pred_path"]
    )

    y_proba = joblib.load(
        config["y_proba_path"]
    )

    return (
        model,
        model_features,
        X_test,
        y_test,
        y_pred,
        y_proba
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Controls"
)


# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.subheader(
    "Model Selection"
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_REGISTRY.keys()),
    key="selected_model"
)

selected_config = MODEL_REGISTRY[
    selected_model_name
]


# ============================================================
# PREDICTION INPUTS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Prediction Inputs"
)

st.sidebar.caption(
    "Enter an individual's information."
)


# ============================================================
# DEMOGRAPHIC & PHYSICAL
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
    [
        "yes",
        "no"
    ],
    key="input_family_history"
)


# ============================================================
# BMI
# ============================================================

bmi = weight / (height ** 2)


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
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
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
    [
        "no",
        "yes"
    ],
    key="input_scc"
)


calc = st.sidebar.selectbox(
    "Alcohol consumption (CALC)",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
    key="input_calc"
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
# MODEL AVAILABILITY
# ============================================================

if not selected_config.get(
    "available",
    False
):

    st.info(
        f"**{selected_model_name}** has not been imported yet."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

(
    model,
    model_features,
    X_test,
    y_test,
    y_pred,
    y_proba
) = load_artifacts(
    selected_config
)


# ============================================================
# VERIFY MODEL FEATURES
# ============================================================

if len(model_features) != len(
    model.feature_importances_
):

    st.error(
        "Model feature count does not match "
        "model_features.pkl."
    )

    st.stop()


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = model.classes_


y_test_binarized = label_binarize(
    y_test,
    classes=class_names
)


# ============================================================
# MODEL PERFORMANCE
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
# CREATE INPUT DATAFRAME
# ============================================================

input_data = {

    "Age":
        age,

    "Height":
        height,

    "Weight":
        weight,

    "FCVC":
        fcvc,

    "NCP":
        ncp,

    "CH2O":
        ch2o,

    "FAF":
        faf,

    "TUE":
        tue,

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


input_aligned = pd.DataFrame(
    [input_data]
)


# ============================================================
# FORCE EXACT MODEL FEATURE ORDER
# ============================================================

input_aligned = input_aligned.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================================
# VERIFY INPUT
# ============================================================

if input_aligned.shape[1] != len(
    model_features
):

    st.error(
        f"Feature mismatch: model expects "
        f"{len(model_features)} features but "
        f"received {input_aligned.shape[1]}."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_aligned
)[0]


prediction_proba = model.predict_proba(
    input_aligned
)[0]


prediction_probability = (
    prediction_proba.max()
)


# ============================================================
# PROBABILITY TABLE
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
# MAIN PERFORMANCE
# ============================================================

st.header(
    f"{selected_model_name} — Performance Overview"
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

st.subheader(
    "Prediction Result"
)


prediction_col1, prediction_col2 = st.columns(
    [1, 2]
)


with prediction_col1:

    st.markdown(
        "### Predicted Obesity Level"
    )

    st.success(
        f"## {prediction}"
    )

    st.metric(
        "Prediction Probability",
        f"{prediction_probability:.2%}"
    )


with prediction_col2:

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
        "Predicted Probability"
    )

    ax.set_xlim(
        0,
        1
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    gc.collect()


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
# MODEL INPUT VERIFICATION
# ============================================================

with st.expander(
    "View Encoded Model Input"
):

    st.write(
        "The following 23 features are sent directly "
        "to the Random Forest model:"
    )

    st.dataframe(
        input_aligned,
        hide_index=True,
        width="stretch"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.header(
    "Model Comparison"
)


def calculate_model_metrics(
    model_name
):

    config = MODEL_REGISTRY[
        model_name
    ]

    if not config.get(
        "available",
        False
    ):

        return {

            "Model":
                model_name,

            "Accuracy":
                "Not Available",

            "F1 (Weighted)":
                "Not Available",

            "Precision (Weighted)":
                "Not Available",

            "Recall (Weighted)":
                "Not Available",

            "ROC-AUC (Macro)":
                "Not Available"
        }


    try:

        (
            loaded_model,
            loaded_features,
            loaded_X_test,
            loaded_y_test,
            loaded_y_pred,
            loaded_y_proba
        ) = load_artifacts(
            config
        )

        loaded_classes = (
            loaded_model.classes_
        )

        loaded_y_test_bin = (
            label_binarize(
                loaded_y_test,
                classes=loaded_classes
            )
        )

        return {

            "Model":
                model_name,

            "Accuracy":
                f"{accuracy_score(
                    loaded_y_test,
                    loaded_y_pred
                ):.2%}",

            "F1 (Weighted)":
                f"{f1_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "Precision (Weighted)":
                f"{precision_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "Recall (Weighted)":
                f"{recall_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "ROC-AUC (Macro)":
                f"{roc_auc_score(
                    loaded_y_test_bin,
                    loaded_y_proba,
                    average='macro',
                    multi_class='ovr'
                ):.4f}"
        }


    except Exception:

        return {

            "Model":
                model_name,

            "Accuracy":
                "Error",

            "F1 (Weighted)":
                "Error",

            "Precision (Weighted)":
                "Error",

            "Recall (Weighted)":
                "Error",

            "ROC-AUC (Macro)":
                "Error"
        }


comparison_rows = []


for model_name in MODEL_REGISTRY:

    comparison_rows.append(
        calculate_model_metrics(
            model_name
        )
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# RESULTS NAVIGATION
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
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

if section == "Confusion Matrix":

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        selected_config[
            "confusion_matrix_image"
        ],
        width="stretch"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

elif section == "Classification Report":

    st.subheader(
        "Classification Report"
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(
            report_dict
        )
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

    st.image(
        selected_config[
            "feature_importance_image"
        ],
        width="stretch"
    )

    if hasattr(
        model,
        "feature_importances_"
    ):

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
            width="stretch",
            hide_index=True
        )


# ============================================================
# ROC CURVES
# ============================================================

elif section == "ROC Curves":

    st.subheader(
        "ROC Curves"
    )

    st.image(
        selected_config[
            "roc_curve_image"
        ],
        width="stretch"
    )

"""
Obesity Level Classification — Model Comparison Dashboard
============================================================
Run with:

    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import gc

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
    layout="wide"
)

st.title(
    "Obesity Level Classification — Model Comparison"
)

st.caption(
    "CRISP-DM Project | Estimation of Obesity Levels Based on "
    "Eating Habits and Physical Condition"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "Random Forest": {

        "available": True,

        "model_path":
            "models/random_forest_model.pkl",

        "model_features_path":
            "data/model_features.pkl",

        "X_test_path":
            "data/X_test_encoded.pkl",

        "y_test_path":
            "data/y_test_flat.pkl",

        "y_pred_path":
            "data/rf_y_pred.pkl",

        "y_proba_path":
            "data/rf_y_pred_proba.pkl",

        "confusion_matrix_image":
            "images/rf_confusion_matrix.png",

        "feature_importance_image":
            "images/rf_feature_importance.png",

        "roc_curve_image":
            "images/rf_roc_curves.png",
    },

    "K-Nearest Neighbours (KNN)": {
        "available": False,
    },

    "Logistic Regression": {
        "available": False,
    },

    "Gradient Boosting": {
        "available": False,
    },
}


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts(config):

    model = joblib.load(
        config["model_path"]
    )

    model_features = joblib.load(
        config["model_features_path"]
    )

    X_test = joblib.load(
        config["X_test_path"]
    )

    y_test = joblib.load(
        config["y_test_path"]
    )

    y_pred = joblib.load(
        config["y_pred_path"]
    )

    y_proba = joblib.load(
        config["y_proba_path"]
    )

    return (
        model,
        model_features,
        X_test,
        y_test,
        y_pred,
        y_proba
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Controls"
)


# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.subheader(
    "Model Selection"
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_REGISTRY.keys()),
    key="selected_model"
)

selected_config = MODEL_REGISTRY[
    selected_model_name
]


# ============================================================
# PREDICTION INPUTS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Prediction Inputs"
)

st.sidebar.caption(
    "Enter an individual's information."
)


# ============================================================
# DEMOGRAPHIC & PHYSICAL
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
    [
        "yes",
        "no"
    ],
    key="input_family_history"
)


# ============================================================
# BMI
# ============================================================

bmi = weight / (height ** 2)


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
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
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
    [
        "no",
        "yes"
    ],
    key="input_scc"
)


calc = st.sidebar.selectbox(
    "Alcohol consumption (CALC)",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ],
    key="input_calc"
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
# MODEL AVAILABILITY
# ============================================================

if not selected_config.get(
    "available",
    False
):

    st.info(
        f"**{selected_model_name}** has not been imported yet."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

(
    model,
    model_features,
    X_test,
    y_test,
    y_pred,
    y_proba
) = load_artifacts(
    selected_config
)


# ============================================================
# VERIFY MODEL FEATURES
# ============================================================

if len(model_features) != len(
    model.feature_importances_
):

    st.error(
        "Model feature count does not match "
        "model_features.pkl."
    )

    st.stop()


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = model.classes_


y_test_binarized = label_binarize(
    y_test,
    classes=class_names
)


# ============================================================
# MODEL PERFORMANCE
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
# CREATE INPUT DATAFRAME
# ============================================================

input_data = {

    "Age":
        age,

    "Height":
        height,

    "Weight":
        weight,

    "FCVC":
        fcvc,

    "NCP":
        ncp,

    "CH2O":
        ch2o,

    "FAF":
        faf,

    "TUE":
        tue,

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


input_aligned = pd.DataFrame(
    [input_data]
)


# ============================================================
# FORCE EXACT MODEL FEATURE ORDER
# ============================================================

input_aligned = input_aligned.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================================
# VERIFY INPUT
# ============================================================

if input_aligned.shape[1] != len(
    model_features
):

    st.error(
        f"Feature mismatch: model expects "
        f"{len(model_features)} features but "
        f"received {input_aligned.shape[1]}."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_aligned
)[0]


prediction_proba = model.predict_proba(
    input_aligned
)[0]


prediction_probability = (
    prediction_proba.max()
)


# ============================================================
# PROBABILITY TABLE
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
# MAIN PERFORMANCE
# ============================================================

st.header(
    f"{selected_model_name} — Performance Overview"
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

st.subheader(
    "Prediction Result"
)


prediction_col1, prediction_col2 = st.columns(
    [1, 2]
)


with prediction_col1:

    st.markdown(
        "### Predicted Obesity Level"
    )

    st.success(
        f"## {prediction}"
    )

    st.metric(
        "Prediction Probability",
        f"{prediction_probability:.2%}"
    )


with prediction_col2:

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
        "Predicted Probability"
    )

    ax.set_xlim(
        0,
        1
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    gc.collect()


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
# MODEL INPUT VERIFICATION
# ============================================================

with st.expander(
    "View Encoded Model Input"
):

    st.write(
        "The following 23 features are sent directly "
        "to the Random Forest model:"
    )

    st.dataframe(
        input_aligned,
        hide_index=True,
        width="stretch"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.header(
    "Model Comparison"
)


def calculate_model_metrics(
    model_name
):

    config = MODEL_REGISTRY[
        model_name
    ]

    if not config.get(
        "available",
        False
    ):

        return {

            "Model":
                model_name,

            "Accuracy":
                "Not Available",

            "F1 (Weighted)":
                "Not Available",

            "Precision (Weighted)":
                "Not Available",

            "Recall (Weighted)":
                "Not Available",

            "ROC-AUC (Macro)":
                "Not Available"
        }


    try:

        (
            loaded_model,
            loaded_features,
            loaded_X_test,
            loaded_y_test,
            loaded_y_pred,
            loaded_y_proba
        ) = load_artifacts(
            config
        )

        loaded_classes = (
            loaded_model.classes_
        )

        loaded_y_test_bin = (
            label_binarize(
                loaded_y_test,
                classes=loaded_classes
            )
        )

        return {

            "Model":
                model_name,

            "Accuracy":
                f"{accuracy_score(
                    loaded_y_test,
                    loaded_y_pred
                ):.2%}",

            "F1 (Weighted)":
                f"{f1_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "Precision (Weighted)":
                f"{precision_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "Recall (Weighted)":
                f"{recall_score(
                    loaded_y_test,
                    loaded_y_pred,
                    average='weighted'
                ):.4f}",

            "ROC-AUC (Macro)":
                f"{roc_auc_score(
                    loaded_y_test_bin,
                    loaded_y_proba,
                    average='macro',
                    multi_class='ovr'
                ):.4f}"
        }


    except Exception:

        return {

            "Model":
                model_name,

            "Accuracy":
                "Error",

            "F1 (Weighted)":
                "Error",

            "Precision (Weighted)":
                "Error",

            "Recall (Weighted)":
                "Error",

            "ROC-AUC (Macro)":
                "Error"
        }


comparison_rows = []


for model_name in MODEL_REGISTRY:

    comparison_rows.append(
        calculate_model_metrics(
            model_name
        )
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# RESULTS NAVIGATION
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
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

if section == "Confusion Matrix":

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        selected_config[
            "confusion_matrix_image"
        ],
        width="stretch"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

elif section == "Classification Report":

    st.subheader(
        "Classification Report"
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(
            report_dict
        )
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

    st.image(
        selected_config[
            "feature_importance_image"
        ],
        width="stretch"
    )

    if hasattr(
        model,
        "feature_importances_"
    ):

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
            width="stretch",
            hide_index=True
        )


# ============================================================
# ROC CURVES
# ============================================================

elif section == "ROC Curves":

    st.subheader(
        "ROC Curves"
    )

    st.image(
        selected_config[
            "roc_curve_image"
        ],
        width="stretch"
    )

st.write("### DIRECT STREAMLIT PREDICTION TEST")

st.write("Height:", height)
st.write("Weight:", weight)
st.write("BMI:", bmi)

st.write("Prediction:", prediction)

for cls, prob in zip(
    model.classes_,
    prediction_proba
):
    st.write(
        f"{cls}: {prob:.4%}"
    )
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Obesity Levels dataset — "
    "UCI Machine Learning Repository"
)


