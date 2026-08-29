"""
BMDS2003 Data Science — Obesity Risk Analytics & Classification Dashboard
=========================================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb  # noqa: F401 — required when loading serialized XGBoost models

from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Obesity Risk Analytics & Classification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PROJECT CONSTANTS
# =============================================================================

TARGET = "NObeyesdad"

NUMERICAL_COLUMNS = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

EXPECTED_CATEGORIES = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["no", "yes"],
    "FAVC": ["no", "yes"],
    "CAEC": ["Always", "Frequently", "Sometimes", "no"],
    "SMOKE": ["no", "yes"],
    "SCC": ["no", "yes"],
    "CALC": ["Always", "Frequently", "Sometimes", "no"],
    "MTRANS": [
        "Automobile",
        "Bike",
        "Motorbike",
        "Public_Transportation",
        "Walking",
    ],
}

# Natural obesity progression used only for presentation/analysis ordering.
# This does NOT change any model's internal class/probability order.
OBESITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

DISPLAY_LABELS = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
}

# XGBoost was trained with integer IDs created by LabelEncoder.
# Keep this EXACTLY aligned with LabelEncoder.classes_ in the established project run.
XGB_CLASS_NAMES = np.array(
    [
        "Insufficient_Weight",
        "Normal_Weight",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
        "Overweight_Level_I",
        "Overweight_Level_II",
    ],
    dtype=str,
)

# Consistent ordinal palette: light = lower category, dark = higher category.
OBESITY_COLORS = {
    "Insufficient_Weight": "#D9F0F0",
    "Normal_Weight": "#B8E0DF",
    "Overweight_Level_I": "#90C9C7",
    "Overweight_Level_II": "#62AAA9",
    "Obesity_Type_I": "#3A8688",
    "Obesity_Type_II": "#25666D",
    "Obesity_Type_III": "#164A57",
}

MODEL_COLORS = {
    "Logistic Regression": "#6B7F93",
    "K-Nearest Neighbours (KNN)": "#5C9EAD",
    "Random Forest": "#3A7D77",
    "XGBoost": "#1F5F75",
}

MODEL_SHORT_LABELS = {
    "Logistic Regression": "Logistic Regression",
    "K-Nearest Neighbours (KNN)": "KNN",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

NUMERIC_UNITS = {
    "Age": "years",
    "Height": "m",
    "Weight": "kg",
    "FCVC": "vegetable-consumption scale (1–3)",
    "NCP": "main meals (1–4)",
    "CH2O": "water-consumption scale (1–3)",
    "FAF": "physical-activity frequency scale (0–3)",
    "TUE": "technology-use scale (0–2)",
}

# Dataset context stated in the written report/UCI description.
DATASET_CONTEXT = {
    "countries": "Mexico, Peru and Colombia",
    "synthetic_share": 0.77,
    "collected_share": 0.23,
}

MODEL_REGISTRY = {
    "Logistic Regression": {
        "model_candidates": [
            "models/logistic_regression_model.pkl",
            "logistic_regression_model.pkl",
            "logistic_regression_model(1).pkl",
        ],
        "evaluation": {
            "X_test": ["data/X_test_encoded.pkl", "X_test_encoded.pkl"],
            "y_test": ["data/y_test_flat.pkl", "y_test_flat.pkl"],
            "y_pred": ["data/lr_y_pred.pkl", "lr_y_pred.pkl"],
            "y_proba": ["data/lr_y_pred_proba.pkl", "lr_y_pred_proba.pkl"],
        },
        "feature_data_candidates": [
            "data/lr_feature_importance.csv",
            "lr_feature_importance.csv",
        ],
    },
    "K-Nearest Neighbours (KNN)": {
        "model_candidates": [
            "models/knn_model.pkl",
            "knn_model.pkl",
            "knn_model (1).pkl",
        ],
        "evaluation": {
            "X_test": ["data/X_test_encoded.pkl", "X_test_encoded.pkl"],
            "y_test": ["data/y_test_flat.pkl", "y_test_flat.pkl"],
            "y_pred": ["data/knn_y_pred.pkl", "knn_y_pred.pkl"],
            "y_proba": ["data/knn_y_pred_proba.pkl", "knn_y_pred_proba.pkl"],
        },
        "feature_data_candidates": [
            "data/knn_feature_importance.csv",
            "knn_feature_importance.csv",
        ],
    },
    "Random Forest": {
        "model_candidates": [
            "models/random_forest_model.pkl",
            "random_forest_model.pkl",
            "random_forest_model (2).pkl",
        ],
        "evaluation": {
            "X_test": ["data/X_test_encoded.pkl", "X_test_encoded.pkl"],
            "y_test": ["data/y_test_flat.pkl", "y_test_flat.pkl"],
            "y_pred": ["data/rf_y_pred.pkl", "rf_y_pred.pkl"],
            "y_proba": ["data/rf_y_pred_proba.pkl", "rf_y_pred_proba.pkl"],
        },
        "feature_data_candidates": [
            "data/rf_feature_importance.csv",
            "rf_feature_importance.csv",
        ],
    },
    "XGBoost": {
        "model_candidates": [
            "models/xgboost_model.pkl",
            "xgboost_model.pkl",
            "xgboost_model (4).pkl",
        ],
        "evaluation": {
            "X_test": ["data/X_test_encoded.pkl", "X_test_encoded.pkl"],
            "y_test": ["data/y_test_flat.pkl", "y_test_flat.pkl"],
            "y_pred": ["data/xgb_y_pred.pkl", "xgb_y_pred.pkl"],
            "y_proba": ["data/xgb_y_pred_proba.pkl", "xgb_y_pred_proba.pkl"],
        },
        "feature_data_candidates": [
            "data/xgb_feature_importance.csv",
            "xgb_feature_importance.csv",
        ],
    },
}

# The accepted project results are used as a compatibility fingerprint only.
# The UI never substitutes these values for live model calculations.
PROJECT_TEST_RESULTS = {
    "Logistic Regression": {
        "Accuracy": 0.9011,
        "Precision": 0.8998,
        "Recall": 0.9011,
        "Weighted F1": 0.9002,
        "Macro F1": 0.8979,
        "ROC-AUC": 0.9865,
        "Errors": 62,
    },
    "K-Nearest Neighbours (KNN)": {
        "Accuracy": 0.8820,
        "Precision": 0.8817,
        "Recall": 0.8820,
        "Weighted F1": 0.8787,
        "Macro F1": 0.8752,
        "ROC-AUC": 0.9706,
        "Errors": 74,
    },
    "Random Forest": {
        "Accuracy": 0.9394,
        "Precision": 0.9434,
        "Recall": 0.9394,
        "Weighted F1": 0.9402,
        "Macro F1": 0.9379,
        "ROC-AUC": 0.9938,
        "Errors": 38,
    },
    "XGBoost": {
        "Accuracy": 0.9649,
        "Precision": 0.9652,
        "Recall": 0.9649,
        "Weighted F1": 0.9650,
        "Macro F1": 0.9640,
        "ROC-AUC": 0.9985,
        "Errors": 22,
    },
}

# Five-fold cross-validation presentation values, keyed strictly by model name.
# The Random Forest variation values intentionally preserve the model's displayed
# convention used in the project presentation output.
CV_RESULTS = {
    "Logistic Regression": {
        "CV Accuracy": 0.8568,
        "CV Accuracy Variation": 0.0214,
        "CV F1": 0.8542,
        "CV F1 Variation": 0.0227,
    },
    "K-Nearest Neighbours (KNN)": {
        "CV Accuracy": 0.8822,
        "CV Accuracy Variation": 0.0148,
        "CV F1": 0.8779,
        "CV F1 Variation": 0.0165,
    },
    "Random Forest": {
        "CV Accuracy": 0.9253,
        "CV Accuracy Variation": 0.0416,
        "CV F1": 0.9264,
        "CV F1 Variation": 0.0421,
    },
    "XGBoost": {
        "CV Accuracy": 0.9651,
        "CV Accuracy Variation": 0.0105,
        "CV F1": 0.9649,
        "CV F1 Variation": 0.0105,
    },
}

PROJECT_TRAIN_ACCURACY = {
    "Logistic Regression": 0.8966,
    "K-Nearest Neighbours (KNN)": 1.0000,
    "Random Forest": 0.9959,
    "XGBoost": 1.0000,
}

MODEL_METRIC_TOLERANCE = 0.0015

MODEL_INFO = {
    "Logistic Regression": {
        "role": "Baseline model",
        "family": "Linear probabilistic classifier",
        "configuration": (
            "max_iter = 2000, random_state = 42. The model pipeline scales the eight numerical predictors with StandardScaler."
        ),
        "strength": "Highly interpretable, fast, and a strong reference baseline.",
        "limitation": "Linear decision boundaries can miss nonlinear interactions.",
        "interpretability": "High",
        "operational_complexity": "Low",
    },
    "K-Nearest Neighbours (KNN)": {
        "role": "Distance-based nonlinear model",
        "family": "Instance-based classifier",
        "configuration": (
            "k = 3, Manhattan distance, distance weighting. The model pipeline applies StandardScaler to the eight numerical predictors."
        ),
        "strength": "Captures local neighbourhood structure without a linear-boundary assumption.",
        "limitation": "Distance-sensitive and prediction cost grows with training-set size.",
        "interpretability": "Medium–Low",
        "operational_complexity": "Medium",
    },
    "Random Forest": {
        "role": "Nonlinear ensemble model",
        "family": "Bagged decision-tree ensemble",
        "configuration": (
            "500 trees, max_depth = 12, min_samples_split = 5, min_samples_leaf = 2, "
            "class_weight = 'balanced'. Uses encoded unscaled predictors."
        ),
        "strength": "Captures nonlinear relationships/interactions and provides tree importance.",
        "limitation": "Less directly interpretable than the baseline and shows a larger train–test gap.",
        "interpretability": "Medium",
        "operational_complexity": "Medium",
    },
    "XGBoost": {
        "role": "Boosted-tree final candidate",
        "family": "Gradient-boosted decision trees",
        "configuration": (
            "learning_rate = 0.1, max_depth = 5, subsample = 1.0. Uses the exact 23-feature "
            "encoded schema and integer target IDs decoded through XGB_CLASS_NAMES."
        ),
        "strength": "Strong predictive performance and captures complex nonlinear interactions.",
        "limitation": "Lower intrinsic interpretability and greater tuning/deployment complexity.",
        "interpretability": "Medium–Low",
        "operational_complexity": "Medium–High",
    },
}


# =============================================================================
# DESIGN SYSTEM
# =============================================================================

def inject_css() -> None:
    """Theme-safe visual layer for both Streamlit Light and Dark mode."""
    st.markdown(
        """
        <style>
        :root {
            --app-primary: var(--primary-color, #2B7A78);
            --app-bg: var(--background-color, #FFFFFF);
            --app-secondary: var(--secondary-background-color, #F0F2F6);
            --app-text: var(--text-color, #31333F);

            --app-panel: var(--app-bg);
            --app-panel-soft: rgba(127, 127, 127, 0.065);
            --app-panel-hover: rgba(127, 127, 127, 0.10);
            --app-border: rgba(127, 127, 127, 0.25);
            --app-shadow: 0 6px 22px rgba(0, 0, 0, 0.08);

            --ink: var(--app-text);
            --muted: color-mix(in srgb, var(--app-text) 67%, var(--app-bg) 33%);
            --panel: var(--app-panel);
            --bg: var(--app-bg);
            --line: var(--app-border);
            --teal: var(--app-primary);
            --teal-dark: color-mix(in srgb, var(--app-primary) 82%, var(--app-text) 18%);
            --teal-soft: color-mix(in srgb, var(--app-primary) 12%, var(--app-bg) 88%);
            --warning: #C58B16;
        }

        /* ===== MOTION / INTERACTION LAYER ===== */
        @keyframes appFadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: none; }
        }

        @keyframes heroGlow {
            0%, 100% { transform: translate3d(0, 0, 0) scale(1); opacity: .45; }
            50% { transform: translate3d(-12px, 8px, 0) scale(1.08); opacity: .72; }
        }

        @keyframes floatOrb {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-10px) rotate(4deg); }
        }

        @keyframes pulseDot {
            0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--app-primary) 28%, transparent); }
            50% { box-shadow: 0 0 0 7px color-mix(in srgb, var(--app-primary) 0%, transparent); }
        }

        @keyframes shine {
            0% { transform: translateX(-130%) skewX(-18deg); }
            55%, 100% { transform: translateX(230%) skewX(-18deg); }
        }

        @keyframes buttonPulse {
            0%, 100% { box-shadow: 0 5px 16px rgba(0,0,0,.08); }
            50% { box-shadow: 0 7px 22px color-mix(in srgb, var(--app-primary) 22%, transparent); }
        }

        .stApp,
        [data-testid="stAppViewContainer"] {
            position: relative;
            overflow-x: visible;
            background:
                radial-gradient(
                    circle at 94% 4%,
                    color-mix(in srgb, var(--app-primary) 7%, transparent),
                    transparent 25rem
                ),
                radial-gradient(
                    circle at 5% 22%,
                    color-mix(in srgb, var(--app-primary) 4%, transparent),
                    transparent 28rem
                ),
                var(--app-bg) !important;
            color: var(--app-text);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            animation: appFadeUp .55s ease;
        }

        /* A very subtle moving atmosphere behind the dashboard. */
        [data-testid="stAppViewContainer"]::before,
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
            filter: blur(3px);
        }

        [data-testid="stAppViewContainer"]::before {
            top: 9%;
            right: -80px;
            background: color-mix(in srgb, var(--app-primary) 7%, transparent);
            animation: floatOrb 8s ease-in-out infinite;
        }

        [data-testid="stAppViewContainer"]::after {
            bottom: 7%;
            left: -100px;
            width: 180px;
            height: 180px;
            background: color-mix(in srgb, var(--app-primary) 4%, transparent);
            animation: floatOrb 10s ease-in-out infinite reverse;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--app-text) !important;
            letter-spacing: -0.02em;
        }

        p, li {
            color: var(--app-text);
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .small-note {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--app-secondary) !important;
            border-right: 1px solid var(--app-border) !important;
        }

        [data-testid="stSidebar"] * {
            font-size: 0.96rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--app-border) !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-left: 3px solid transparent;
            border-radius: 9px;
            transition:
                background-color 180ms ease,
                border-color 180ms ease,
                transform 180ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: color-mix(
                in srgb,
                var(--app-primary) 10%,
                transparent
            ) !important;
            transform: translateX(2px);
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) {
            animation: appFadeUp .25s ease both;
            background: color-mix(
                in srgb,
                var(--app-primary) 14%,
                transparent
            ) !important;
            border-left-color: var(--app-primary) !important;
            font-weight: 700;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 30px 32px;
            border: 1px solid var(--app-border);
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    var(--app-panel) 0%,
                    color-mix(
                        in srgb,
                        var(--app-secondary) 78%,
                        var(--app-panel) 22%
                    ) 100%
                );
            box-shadow: var(--app-shadow);
            margin-bottom: 18px;
            animation: appFadeUp .65s ease both;
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 190px;
            height: 190px;
            right: -55px;
            top: -85px;
            border-radius: 50%;
            background: color-mix(in srgb, var(--app-primary) 15%, transparent);
            filter: blur(2px);
            animation: heroGlow 6s ease-in-out infinite;
        }

        .hero::after {
            content: "";
            position: absolute;
            left: -20%;
            top: 0;
            width: 35%;
            height: 100%;
            background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--app-primary) 8%, transparent), transparent);
            transform: translateX(-130%) skewX(-18deg);
            animation: shine 8s ease-in-out infinite;
            pointer-events: none;
        }

        .hero h1, .hero p {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 2.05rem;
            color: var(--app-text) !important;
        }

        .hero p {
            margin: 0;
            color: var(--muted) !important;
            font-size: 1.02rem;
        }

        .kpi-card {
            position: relative;
            overflow: hidden;
            background: var(--app-panel);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 16px 17px;
            min-height: 112px;
            box-shadow: var(--app-shadow);
            transition:
                transform 220ms ease,
                border-color 220ms ease,
                background-color 220ms ease,
                box-shadow 220ms ease;
            animation: appFadeUp .55s ease both;
        }

        .kpi-card::after {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            width: 34%;
            left: -45%;
            background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--app-primary) 12%, transparent), transparent);
            transform: skewX(-18deg);
            transition: none;
        }

        .kpi-card:hover::after {
            animation: shine 1.05s ease;
        }

        .kpi-card:hover {
            transform: translateY(-1px);
            background: var(--app-panel-hover);
            border-color: color-mix(
                in srgb,
                var(--app-primary) 45%,
                var(--app-border) 55%
            );
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 6px;
        }

        .kpi-value {
            color: var(--app-text);
            font-size: 1.58rem;
            font-weight: 760;
            line-height: 1.15;
        }

        .kpi-note {
            color: var(--muted);
            font-size: 0.75rem;
            margin-top: 6px;
        }

        .insight-box {
            background: var(--app-panel-soft);
            border: 1px solid var(--app-border);
            border-left: 4px solid var(--app-primary);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 14px 0;
        }

        .insight-title {
            color: var(--app-text);
            font-weight: 700;
            margin-bottom: 4px;
        }

        .insight-body {
            color: var(--muted);
            line-height: 1.55;
        }

        .callout {
            background: color-mix(
                in srgb,
                var(--app-primary) 11%,
                var(--app-bg) 89%
            );
            border: 1px solid color-mix(
                in srgb,
                var(--app-primary) 30%,
                var(--app-border) 70%
            );
            border-radius: 12px;
            padding: 15px 17px;
            margin: 10px 0;
            color: var(--app-text);
        }

        .flow-step {
            background: var(--app-panel);
            border: 1px solid var(--app-border);
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0;
            display: flex;
            gap: 14px;
            align-items: flex-start;
        }

        .flow-num {
            min-width: 32px;
            height: 32px;
            border-radius: 50%;
            background: color-mix(
                in srgb,
                var(--app-primary) 18%,
                var(--app-bg) 82%
            );
            color: var(--app-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }

        .flow-content b {
            color: var(--app-text);
        }

        .flow-content span {
            color: var(--muted);
            font-size: .92rem;
        }

        .prediction-heading {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 1.35rem;
            font-weight: 760;
            margin: 18px 0 10px;
            animation: appFadeUp .45s ease both;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--app-primary);
            animation: pulseDot 1.8s ease-in-out infinite;
        }

        .section-eyebrow {
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--app-primary);
            font-size: .75rem;
            font-weight: 750;
            margin-bottom: 4px;
        }

        .source-pill {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            border: 1px solid var(--app-border);
            background: var(--app-panel-soft);
            color: var(--muted);
            font-size: .75rem;
            margin-right: 5px;
        }

        div[data-testid="stMetric"] {
            background: var(--app-panel);
            border: 1px solid var(--app-border);
            padding: 12px 14px;
            border-radius: 12px;
            box-shadow: var(--app-shadow);
            transition:
                transform 180ms ease,
                border-color 180ms ease,
                background-color 180ms ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-1px);
            background: var(--app-panel-hover);
            border-color: color-mix(
                in srgb,
                var(--app-primary) 45%,
                var(--app-border) 55%
            );
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--app-text) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: var(--app-panel-soft);
            border-radius: 10px;
            padding: 2px 4px;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--muted) !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--app-text) !important;
            background: var(--app-panel-hover);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--app-text) !important;
            font-weight: 700;
            background: color-mix(
                in srgb,
                var(--app-primary) 11%,
                transparent
            );
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--app-primary) !important;
        }

        [data-testid="stExpander"] {
            border-color: var(--app-border) !important;
            background: transparent !important;
        }

        [data-testid="stExpander"] summary:hover {
            background: var(--app-panel-hover) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-color: var(--app-border) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            position: relative;
            overflow: hidden;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--app-primary) !important;
            transform: translateY(-2px);
            animation: buttonPulse 1.2s ease-in-out infinite;
        }

        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: translateY(0) scale(.985);
        }

        .stButton > button::after,
        .stDownloadButton > button::after {
            content: "";
            position: absolute;
            top: -20%;
            left: -80%;
            width: 35%;
            height: 140%;
            background: rgba(255,255,255,.20);
            transform: skewX(-18deg);
            pointer-events: none;
        }

        .stButton > button:hover::after,
        .stDownloadButton > button:hover::after {
            animation: shine .9s ease;
        }

        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        hr {
            border: none !important;
            border-top: 1px solid var(--app-border) !important;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 22px 20px;
            }

            .hero h1 {
                font-size: 1.7rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: .001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .001ms !important;
                scroll-behavior: auto !important;
            }
        }


        /* =========================================================
           SYSTEM SIDEBAR — LIGHT MODE
           Sidebar presentation only; navigation logic is unchanged.
           ========================================================= */

        [data-testid="stSidebar"] {
            background: #F2F2F7 !important;
            border-right: 1px solid rgba(60, 60, 67, 0.10) !important;
        }

        [data-testid="stSidebar"] > div {
            background: transparent !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1D1D1F !important;
            letter-spacing: -0.015em;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #3A3A3C;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #8E8E93 !important;
        }

        [data-testid="stSidebar"] hr {
            border: 0 !important;
            border-top: 1px solid rgba(60, 60, 67, 0.10) !important;
            margin: 1rem 0 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 3px;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 42px;
            padding: 0.48rem 0.70rem !important;
            margin: 2px 0 !important;
            display: flex !important;
            align-items: center !important;
            border-radius: 9px !important;
            border-left: 3px solid transparent !important;
            background: transparent !important;
            color: #3A3A3C !important;
            transition:
                background-color 150ms ease,
                color 150ms ease,
                border-color 150ms ease !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label span {
            color: #3A3A3C !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(0, 0, 0, 0.045) !important;
            border-left-color: rgba(0, 122, 255, 0.30) !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover span {
            color: #1D1D1F !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(0, 122, 255, 0.10) !important;
            border-left-color: #007AFF !important;
            box-shadow: none !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
            color: #1D1D1F !important;
            font-weight: 650 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] {
            border-radius: 9px !important;
            border-color: rgba(60, 60, 67, 0.16) !important;
            background: rgba(255, 255, 255, 0.72) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
            border-color: #007AFF !important;
            box-shadow: 0 0 0 1px rgba(0, 122, 255, 0.22) !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
        }


        /* =========================================================
           UNIFORM MODEL / RANK CARDS
           ========================================================= */

        .kpi-card {
            min-height: 160px !important;
            height: 160px !important;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
        }

        .kpi-label {
            min-height: 1.3rem;
        }

        .kpi-value {
            min-height: 3.3rem;
            display: flex;
            align-items: center;
            overflow-wrap: anywhere;
        }

        .kpi-note {
            margin-top: auto !important;
            min-height: 1.8rem;
        }

        .rank-card {
            height: 184px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            padding: 15px 16px;
            border-radius: 14px;
            border: 1px solid var(--app-border);
            background: var(--app-panel);
            box-shadow: var(--app-shadow);
            transition: transform 180ms ease, border-color 180ms ease, background-color 180ms ease;
        }

        .rank-card:hover {
            transform: translateY(-1px);
            background: var(--app-panel-hover);
            border-color: color-mix(in srgb, var(--app-primary) 38%, var(--app-border) 62%);
        }

        .rank-card-best {
            border-color: color-mix(in srgb, var(--app-primary) 62%, var(--app-border) 38%);
        }

        .rank-card-rank {
            color: var(--muted);
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            min-height: 1.1rem;
        }

        .rank-card-title {
            color: var(--app-text);
            font-size: .96rem;
            font-weight: 700;
            line-height: 1.28;
            min-height: 2.7rem;
            max-height: 2.7rem;
            display: flex;
            align-items: flex-start;
            overflow: hidden;
            margin-top: 5px;
        }

        .rank-card-value {
            color: var(--app-text);
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 760;
            min-height: 2.2rem;
            display: flex;
            align-items: center;
            margin-top: 6px;
        }

        .rank-card-gap {
            color: var(--muted);
            font-size: .78rem;
            margin-top: auto;
            min-height: 1.15rem;
        }

        /* =========================================================
           SIDEBAR PROJECT CONTEXT
           ========================================================= */

        .sidebar-mini-card {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(60, 60, 67, 0.10);
            border-radius: 12px;
            padding: 12px;
            margin: 8px 0 10px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
        }

        .sidebar-mini-title {
            color: #1D1D1F;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .01em;
            margin-bottom: 8px;
        }

        .sidebar-snapshot-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .sidebar-snapshot-cell {
            padding: 7px 8px;
            border-radius: 9px;
            background: rgba(255,255,255,.52);
            border: 1px solid rgba(60,60,67,.07);
        }

        .sidebar-snapshot-value {
            color: #1D1D1F;
            font-size: 1.05rem;
            font-weight: 760;
            line-height: 1.1;
        }

        .sidebar-snapshot-label,
        .sidebar-context-line,
        .sidebar-active-metric,
        .sidebar-dataset-line {
            color: #8E8E93;
            font-size: .69rem;
            line-height: 1.3;
        }

        .sidebar-active-name {
            color: #1D1D1F;
            font-size: .88rem;
            font-weight: 700;
            line-height: 1.3;
            margin: 3px 0 7px 0;
        }

        .sidebar-active-row {
            display: flex;
            align-items: center;
            gap: 7px;
        }

        .sidebar-active-dot {
            width: 8px;
            height: 8px;
            flex: 0 0 8px;
            border-radius: 50%;
        }

        .sidebar-context-box {
            padding: 3px 2px 7px 2px;
        }

        .sidebar-dataset {
            padding: 3px 2px 0 2px;
        }

        /* Plot containers remain unconstrained so chart controls are not clipped. */
        [data-testid="stPlotlyChart"],
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            overflow: visible !important;
        }

        [data-testid="stPlotlyChart"] .modebar {
            top: 8px !important;
            right: 8px !important;
            z-index: 20 !important;
        }

        button[title="View fullscreen"],
        button[title="Exit fullscreen"] {
            z-index: 30 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# =============================================================================
# Dark Mode ONLY — Light Mode remains exactly as defined above.
# =============================================================================

def inject_dark_mode() -> None:
    """
    This block runs only when Streamlit's active appearance is Dark.
    It does not modify layout, analytics, models, data, prediction logic,
    navigation, session state, charts' data, forms, or Light Mode styling.
    """
    try:
        is_dark = str(st.context.theme.type).lower() == "dark"
    except Exception:
        is_dark = False

    if not is_dark:
        return

    st.markdown(
        """
        <style>

        /* =========================================================
            DARK PALETTE
           ========================================================= */

        :root {
            --primary-color: #0A84FF;

            --app-primary: #0A84FF;

            --app-bg: #1C1C1E;
            --app-secondary: #242426;
            --app-panel: #2C2C2E;
            --app-panel-soft: #242426;
            --app-panel-hover: #323234;

            --app-text: #F5F5F7;

            --app-border: rgba(255, 255, 255, 0.08);

            --app-shadow:
                0 1px 2px rgba(0, 0, 0, 0.25),
                0 8px 24px rgba(0, 0, 0, 0.12);

            --ink: #F5F5F7;
            --muted: #A1A1A6;

            --panel: #2C2C2E;
            --bg: #1C1C1E;
            --line: rgba(255, 255, 255, 0.08);

            --teal: #0A84FF;
            --teal-dark: #409CFF;
            --teal-soft: rgba(10, 132, 255, 0.10);

            --warning: #FF9F0A;

            color-scheme: dark;
        }


        /* =========================================================
           MAIN APPLICATION SURFACE
           ========================================================= */

        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #1C1C1E !important;
            color: #F5F5F7 !important;
        }

        /* Keep existing atmosphere but remove colourful/glowing effect. */
        [data-testid="stAppViewContainer"]::before,
        [data-testid="stAppViewContainer"]::after {
            background: rgba(255, 255, 255, 0.008) !important;
            filter: blur(12px) !important;
        }


        /* =========================================================
           TYPOGRAPHY
           ========================================================= */

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
            color: #F5F5F7 !important;
            text-shadow: none !important;
        }

        p,
        li {
            color: #F5F5F7;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .small-note,
        .kpi-label,
        .kpi-note,
        .flow-content span {
            color: #A1A1A6 !important;
        }

        .section-eyebrow {
            color: #0A84FF !important;
        }


        /* =========================================================
           SIDEBAR — SYSTEM STYLE
           ========================================================= */

        [data-testid="stSidebar"] {
            background: #242426 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #F5F5F7;
        }

        [data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] p {
            color: #8E8E93 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08) !important;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label {
            border-left: 3px solid transparent !important;
            border-radius: 9px;
            background: transparent !important;
            transition:
                background-color 180ms ease,
                border-color 180ms ease,
                transform 180ms ease;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:hover {
            background: #303032 !important;
            border-left-color: rgba(10, 132, 255, 0.35) !important;
            transform: translateX(1px);
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) {
            background: #3A3A3C !important;
            border-left-color: #0A84FF !important;
            font-weight: 600 !important;
        }


        /* =========================================================
           HERO
           ========================================================= */

        .hero {
            background: #242426 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow:
                0 1px 2px rgba(0, 0, 0, 0.22),
                0 8px 24px rgba(0, 0, 0, 0.10) !important;
        }

        .hero h1 {
            color: #F5F5F7 !important;
        }

        .hero p {
            color: #A1A1A6 !important;
        }

        .hero::before {
            background: rgba(255, 255, 255, 0.018) !important;
            filter: blur(10px) !important;
        }

        .hero::after {
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.025),
                    transparent
                ) !important;
        }


        /* =========================================================
           KPI / CUSTOM CARDS
           ========================================================= */

        .kpi-card {
            background: #2C2C2E !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow:
                0 1px 2px rgba(0, 0, 0, 0.25),
                0 8px 24px rgba(0, 0, 0, 0.12) !important;
        }

        .kpi-card:hover {
            background: #323234 !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
            transform: translateY(-1px) !important;
            box-shadow:
                0 1px 3px rgba(0, 0, 0, 0.26),
                0 9px 26px rgba(0, 0, 0, 0.13) !important;
        }

        .kpi-card::after {
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.035),
                    transparent
                ) !important;
        }

        .kpi-value {
            color: #F5F5F7 !important;
        }

        .kpi-label {
            color: #A1A1A6 !important;
        }

        .kpi-note {
            color: #8E8E93 !important;
        }


        /* =========================================================
           STREAMLIT METRIC CARDS
           ========================================================= */

        div[data-testid="stMetric"] {
            background: #2C2C2E !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow:
                0 1px 2px rgba(0, 0, 0, 0.25),
                0 8px 24px rgba(0, 0, 0, 0.10) !important;
        }

        div[data-testid="stMetric"]:hover {
            background: #323234 !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stMetric"]
        [data-testid="stMetricValue"] {
            color: #F5F5F7 !important;
        }

        div[data-testid="stMetric"]
        [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] label {
            color: #A1A1A6 !important;
        }


        /* =========================================================
           ANALYTICAL INSIGHT BOXES
           ========================================================= */

        .insight-box {
            background: rgba(10, 132, 255, 0.085) !important;
            border: 1px solid rgba(10, 132, 255, 0.20) !important;
            border-left: 3px solid #0A84FF !important;
            box-shadow: none !important;
        }

        .insight-title {
            color: #F5F5F7 !important;
        }

        .insight-body {
            color: #A1A1A6 !important;
        }

        .callout {
            background: rgba(10, 132, 255, 0.08) !important;
            border: 1px solid rgba(10, 132, 255, 0.18) !important;
            color: #F5F5F7 !important;
            box-shadow: none !important;
        }


        /* =========================================================
           DATA PREPARATION FLOW
           ========================================================= */

        .flow-step {
            background: #2C2C2E !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: none !important;
        }

        .flow-content b {
            color: #F5F5F7 !important;
        }

        .flow-content span {
            color: #A1A1A6 !important;
        }

        .flow-num {
            background: rgba(10, 132, 255, 0.13) !important;
            color: #409CFF !important;
            border: 1px solid rgba(10, 132, 255, 0.20);
        }


        /* =========================================================
           SOURCE PILLS
           ========================================================= */

        .source-pill {
            background: #2C2C2E !important;
            color: #A1A1A6 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
        }


        /* =========================================================
           PREDICTION STATUS
           ========================================================= */

        .prediction-heading {
            color: #F5F5F7 !important;
        }

        .status-dot {
            background: #0A84FF !important;
        }


        /* =========================================================
           FORMS
           ========================================================= */

        [data-testid="stForm"] {
            background: #242426 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }


        /* =========================================================
           INPUTS / SELECTBOXES
           ========================================================= */

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stTextInput"] [data-baseweb="input"] {
            background: #2C2C2E !important;
            border-color: rgba(255, 255, 255, 0.10) !important;
            color: #F5F5F7 !important;
            box-shadow: none !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="input"] input {
            color: #F5F5F7 !important;
        }

        [data-baseweb="select"] > div:hover,
        [data-baseweb="input"]:hover {
            border-color: rgba(255, 255, 255, 0.16) !important;
        }

        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"]:focus-within {
            border-color: #0A84FF !important;
            box-shadow: 0 0 0 1px rgba(10, 132, 255, 0.35) !important;
        }

        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [role="listbox"] {
            background: #2C2C2E !important;
            color: #F5F5F7 !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.30) !important;
        }

        [role="option"] {
            color: #F5F5F7 !important;
        }

        [role="option"]:hover {
            background: #3A3A3C !important;
        }

        [role="option"][aria-selected="true"] {
            background: rgba(10, 132, 255, 0.15) !important;
        }

        [data-baseweb="slider"] [role="slider"] {
            background: #F5F5F7 !important;
            border: 2px solid #0A84FF !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.28) !important;
        }


        /* =========================================================
           TABS
           ========================================================= */

        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            color: #8E8E93 !important;
            background: transparent !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #D1D1D6 !important;
            background: transparent !important;
        }

        .stTabs
        [data-baseweb="tab"][aria-selected="true"] {
            color: #F5F5F7 !important;
            background: transparent !important;
            font-weight: 600 !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #0A84FF !important;
            height: 2px !important;
        }


        /* =========================================================
           EXPANDERS
           ========================================================= */

        [data-testid="stExpander"] {
            background: #242426 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stExpander"] summary {
            color: #F5F5F7 !important;
        }

        [data-testid="stExpander"] summary:hover {
            background: #2C2C2E !important;
        }


        /* =========================================================
           PRIMARY BUTTONS
           ========================================================= */

        [data-testid="stFormSubmitButton"] button,
        [data-testid="stBaseButton-primary"] {
            background: #0A84FF !important;
            border: 1px solid #0A84FF !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border-radius: 9px !important;
            box-shadow: none !important;
        }

        [data-testid="stFormSubmitButton"] button *,
        [data-testid="stBaseButton-primary"] * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #409CFF !important;
            border-color: #409CFF !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
        }


        /* =========================================================
           SECONDARY BUTTONS
           ========================================================= */

        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-tertiary"] {
            background: #3A3A3C !important;
            color: #F5F5F7 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 9px !important;
            box-shadow: none !important;
        }

        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stBaseButton-tertiary"]:hover {
            background: #48484A !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
            color: #F5F5F7 !important;
            transform: translateY(-1px) !important;
        }

        .stButton > button::after,
        .stDownloadButton > button::after,
        [data-testid="stFormSubmitButton"] button::after {
            background: rgba(255, 255, 255, 0.055) !important;
        }


        /* =========================================================
           ALERTS / TABLES / CHART CONTAINERS
           ========================================================= */

        [data-testid="stAlert"] {
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stDataFrame"] {
            background: #242426 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stPlotlyChart"] {
            background: transparent !important;
        }

        hr {
            border: none !important;
            border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
        }


        /* =========================================================
           SOFTEN EXISTING DARK-MODE ANIMATIONS
           Existing animation logic remains unchanged.
           ========================================================= */

        @keyframes buttonPulse {
            0%,
            100% {
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.24);
            }

            50% {
                box-shadow: 0 2px 7px rgba(0, 0, 0, 0.28);
            }
        }

        @keyframes pulseDot {
            0%,
            100% {
                box-shadow: 0 0 0 0 rgba(10, 132, 255, 0.18);
            }

            50% {
                box-shadow: 0 0 0 5px rgba(10, 132, 255, 0);
            }
        }


        .sidebar-mini-card {
            background: #2C2C2E !important;
            border-color: rgba(255,255,255,.08) !important;
            box-shadow: none !important;
        }

        .sidebar-snapshot-cell {
            background: #323234 !important;
            border-color: rgba(255,255,255,.06) !important;
        }

        .sidebar-mini-title,
        .sidebar-snapshot-value,
        .sidebar-active-name {
            color: #F5F5F7 !important;
        }

        .sidebar-snapshot-label,
        .sidebar-context-line,
        .sidebar-active-metric,
        .sidebar-dataset-line {
            color: #8E8E93 !important;
        }

        /* =========================================================
           DARK MODE TYPOGRAPHY COLOUR HIERARCHY
           Text colours ONLY — no layout, surface or functionality changes.
           ========================================================= */

        :root {
            --text-primary: #F5F5F7;
            --text-secondary: #A1A1A6;
            --text-muted: #8E8E93;

            --text-blue: #0A84FF;
            --text-cyan: #64D2FF;
            --text-green: #30D158;
            --text-amber: #FF9F0A;
            --text-purple: #BF5AF2;

            --model-logistic: #78B7E8;
            --model-knn: #64D2C8;
            --model-rf: #63D391;
            --model-xgb: #BF7AF0;
        }

        /* Primary hierarchy */
        h1,
        .hero h1 {
            color: var(--text-primary) !important;
        }

        h2 {
            color: var(--text-primary) !important;
        }

        h3,
        h4 {
            color: #D7D7DC !important;
        }

        h5,
        h6 {
            color: var(--text-secondary) !important;
        }

        /* Body copy */
        [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }

        [data-testid="stMarkdownContainer"] li {
            color: var(--text-secondary);
        }

        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b {
            color: var(--text-primary);
        }

        /* Captions and helper text */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .small-note,
        .kpi-note {
            color: var(--text-muted) !important;
        }

        .source-pill {
            color: var(--text-secondary) !important;
        }

        .hero p {
            color: var(--text-secondary) !important;
        }

        /* KPI hierarchy */
        .kpi-label {
            color: var(--text-secondary) !important;
        }

        .kpi-value {
            color: var(--text-primary) !important;
        }

        div[data-testid="stMetric"]
        [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
        }

        div[data-testid="stMetric"]
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        div[data-testid="stMetricDelta"] {
            color: var(--text-muted) !important;
        }

        /* Data preparation */
        .flow-content b {
            color: #79BCFF !important;
        }

        .flow-content span {
            color: var(--text-secondary) !important;
        }

        .flow-num {
            color: var(--text-cyan) !important;
        }

        /* Insight boxes */
        .insight-title {
            color: var(--text-cyan) !important;
        }

        .insight-body {
            color: var(--text-secondary) !important;
        }

        .callout {
            color: var(--text-secondary) !important;
        }

        .callout strong,
        .callout b {
            color: var(--text-blue) !important;
        }

        /* Prediction */
        .prediction-heading {
            color: #79BCFF !important;
        }

        .prediction-heading + div .kpi-value {
            color: var(--text-primary) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-primary) !important;
        }

        [data-testid="stSidebar"] p {
            color: var(--text-secondary) !important;
        }

        [data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] p {
            color: var(--text-muted) !important;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label {
            color: var(--text-secondary) !important;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label span {
            color: var(--text-secondary) !important;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:hover,
        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:hover span {
            color: #D1D1D6 !important;
        }

        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked),
        [data-testid="stSidebar"]
        div[role="radiogroup"]
        label:has(input:checked) span {
            color: var(--text-primary) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            color: var(--text-muted) !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #D1D1D6 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #79BCFF !important;
        }

        /* Form / input labels */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSlider"] label,
        [data-testid="stRadio"] label {
            color: #D1D1D6 !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="input"] input {
            color: var(--text-primary) !important;
        }

        [role="option"] {
            color: #D1D1D6 !important;
        }

        [role="option"][aria-selected="true"] {
            color: var(--text-primary) !important;
        }

        /* Expanders */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {
            color: #D1D1D6 !important;
        }

        [data-testid="stExpander"]
        [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }

        /* Native alerts — text only */
        [data-testid="stAlert"] p {
            color: #D1D1D6 !important;
        }

        [data-testid="stAlert"] strong,
        [data-testid="stAlert"] b {
            color: var(--text-primary) !important;
        }

        /* Data Preparation headings */
        h3[id*="preprocessing"],
        h3[id*="train-test"],
        h3[id*="data-quality"],
        h3[id*="categorical-consistency"],
        h3[id*="outlier"],
        h3[id*="encoding"],
        h3[id*="scaling"] {
            color: #79BCFF !important;
        }

        /* EDA headings */
        h3[id*="obesity-class-distribution"],
        h3[id*="numerical-distribution"],
        h3[id*="obesity-level-vs"],
        h3[id*="weight-height"],
        h3[id*="lifestyle"],
        h3[id*="relationship-correlation"],
        h3[id*="analytical-insights"],
        h3[id*="from-eda"] {
            color: var(--text-cyan) !important;
        }

        /* Modelling / advanced analytics headings */
        h3[id*="model-ranking"],
        h3[id*="performance-comparison"],
        h3[id*="cross-validation"],
        h3[id*="misclassification"],
        h3[id*="strengths-limitations"],
        h3[id*="feature-importance"],
        h3[id*="coefficient"],
        h3[id*="permutation"],
        h3[id*="roc"] {
            color: #C991F4 !important;
        }

        /* Recommendation */
        h3[id*="final-model-recommendation"] {
            color: var(--text-green) !important;
        }

        /* Model heading accents */
        h3[id*="logistic-regression"] {
            color: var(--model-logistic) !important;
        }

        h3[id*="k-nearest-neighbours"],
        h3[id*="knn"] {
            color: var(--model-knn) !important;
        }

        h3[id*="random-forest"] {
            color: var(--model-rf) !important;
        }

        h3[id*="xgboost"] {
            color: var(--model-xgb) !important;
        }

        /* Semantic headings */
        h3[id*="limitation"],
        h4[id*="limitation"],
        h3[id*="dataset-limitation"] {
            color: var(--text-amber) !important;
        }

        h3[id*="prediction"] {
            color: #79BCFF !important;
        }

        /* Footer and links */
        .stApp > footer,
        footer {
            color: var(--text-muted) !important;
        }

        a {
            color: #64A9FF !important;
        }

        a:hover {
            color: #8DC1FF !important;
        }

        /* Button text only */
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button {
            color: #FFFFFF !important;
        }

        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-tertiary"] {
            color: var(--text-primary) !important;
        }

        button:disabled,
        button:disabled * {
            color: #6E6E73 !important;
        }



        /* =========================================================
           SYSTEM SIDEBAR — DARK MODE
           Sidebar presentation only; navigation logic is unchanged.
           ========================================================= */

        [data-testid="stSidebar"] {
            background: #242426 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        }

        [data-testid="stSidebar"] > div {
            background: transparent !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F5F5F7 !important;
        }

        [data-testid="stSidebar"] p {
            color: #A1A1A6 !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #8E8E93 !important;
        }

        [data-testid="stSidebar"] hr {
            border: 0 !important;
            border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
            margin: 1rem 0 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 3px;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 42px;
            padding: 0.48rem 0.70rem !important;
            margin: 2px 0 !important;
            display: flex !important;
            align-items: center !important;
            border-radius: 9px !important;
            border-left: 3px solid transparent !important;
            background: transparent !important;
            transition:
                background-color 150ms ease,
                color 150ms ease,
                border-color 150ms ease !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label span {
            color: #A1A1A6 !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.055) !important;
            border-left-color: rgba(10, 132, 255, 0.40) !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover span {
            color: #D1D1D6 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(255, 255, 255, 0.08) !important;
            border-left-color: #0A84FF !important;
            box-shadow: none !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
            color: #F5F5F7 !important;
            font-weight: 650 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] {
            border-radius: 9px !important;
            border-color: rgba(255, 255, 255, 0.10) !important;
            background: #2C2C2E !important;
            color: #F5F5F7 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
            border-color: #0A84FF !important;
            box-shadow: 0 0 0 1px rgba(10, 132, 255, 0.28) !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_dark_mode()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def first_existing_path(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def display_label(class_name: str) -> str:
    return DISPLAY_LABELS.get(str(class_name), str(class_name).replace("_", " "))


def humanize_category(value: str) -> str:
    return str(value).replace("_", " ")


def fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:.{digits}%}"


def fmt_num(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:.{digits}f}"


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_grid(
    items: List[Tuple[str, str, str]],
    columns_per_row: int = 4,
) -> None:
    for start in range(0, len(items), columns_per_row):
        row = items[start : start + columns_per_row]
        cols = st.columns(columns_per_row)
        for col, item in zip(cols, row):
            label, value, note = item
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-note">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_rank_card(
    rank: int,
    model_name: str,
    score: str,
    gap: str,
    is_best: bool = False,
) -> None:
    """Render one fixed-height ranking card so long model names never shift values."""
    best_class = " rank-card-best" if is_best else ""
    display_name = (
        "K-Nearest Neighbours<br>(KNN)"
        if model_name == "K-Nearest Neighbours (KNN)"
        else model_name
    )
    st.markdown(
        f"""
        <div class="rank-card{best_class}">
            <div class="rank-card-rank">Rank #{rank}</div>
            <div class="rank-card-title">{display_name}</div>
            <div class="rank-card-value">{score}</div>
            <div class="rank-card-gap">Gap from best: {gap}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main_page_changed(current_page: str) -> bool:
    """Return True once when the user changes the main navigation page."""
    previous = st.session_state.get("_previous_main_page")
    if previous is None:
        st.session_state["_previous_main_page"] = current_page
        return False
    if previous != current_page:
        st.session_state["_previous_main_page"] = current_page
        return True
    return False


def scroll_main_view_to_top() -> None:
    """Scroll the parent Streamlit page to the top after a main-page change only."""
    components.html(
        """
        <script>
        (() => {
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const candidates = [
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('section.main')
            ].filter(Boolean);

            for (const element of candidates) {
                if (typeof element.scrollTo === 'function') {
                    element.scrollTo({top: 0, left: 0, behavior: 'auto'});
                }
                element.scrollTop = 0;
            }
            parentWindow.scrollTo({top: 0, left: 0, behavior: 'auto'});
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_sidebar_snapshot(clean: Optional[pd.DataFrame]) -> None:
    records = len(clean) if clean is not None else 2087
    st.sidebar.markdown(
        f"""
        <div class="sidebar-mini-card">
            <div class="sidebar-mini-title">Project Snapshot</div>
            <div class="sidebar-snapshot-grid">
                <div class="sidebar-snapshot-cell">
                    <div class="sidebar-snapshot-value">{records:,}</div>
                    <div class="sidebar-snapshot-label">Clean records</div>
                </div>
                <div class="sidebar-snapshot-cell">
                    <div class="sidebar-snapshot-value">16</div>
                    <div class="sidebar-snapshot-label">Predictors</div>
                </div>
                <div class="sidebar-snapshot-cell">
                    <div class="sidebar-snapshot-value">7</div>
                    <div class="sidebar-snapshot-label">Target classes</div>
                </div>
                <div class="sidebar-snapshot-cell">
                    <div class="sidebar-snapshot-value">4</div>
                    <div class="sidebar-snapshot-label">ML models</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_sidebar_model(page: str) -> Optional[str]:
    if page == "🧪 Model Evaluation":
        return st.session_state.get("model_eval_selector", "Logistic Regression")
    if page == "🔮 Prediction":
        return st.session_state.get("prediction_model", "Logistic Regression")
    return None


def render_sidebar_active_model(page: str) -> None:
    model_name = selected_sidebar_model(page)
    if model_name is None or model_name not in PROJECT_TEST_RESULTS:
        return
    metrics = PROJECT_TEST_RESULTS[model_name]
    accent = MODEL_COLORS.get(model_name, "#0A84FF")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-mini-card">
            <div class="sidebar-mini-title">Active Model</div>
            <div class="sidebar-active-row">
                <span class="sidebar-active-dot" style="background:{accent};"></span>
                <div class="sidebar-active-name">{model_name}</div>
            </div>
            <div class="sidebar-active-metric">Weighted F1&nbsp;&nbsp; {metrics['Weighted F1']:.4f}</div>
            <div class="sidebar-active-metric">Accuracy&nbsp;&nbsp; {metrics['Accuracy']:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_context(page: str, navigation: List[str]) -> None:
    contexts = {
        "🏠 Overview": "Project summary & key findings",
        "⚙️ Data Preparation": "Cleaning & transformation",
        "🔍 Exploratory Analysis": "Descriptive analytics & relationships",
        "🧪 Model Evaluation": "Performance by model",
        "🏆 Model Comparison": "Compare all classifiers",
        "🔮 Prediction": "Interactive classification",
    }
    position = navigation.index(page) + 1
    short_page = page.split(" ", 1)[1] if " " in page else page
    st.sidebar.markdown(
        f"""
        <div class="sidebar-context-box">
            <div class="sidebar-context-line">{position} / {len(navigation)} · {short_page}</div>
            <div class="sidebar-context-line">{contexts.get(page, '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_dataset_source() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-dataset">
            <div class="sidebar-mini-title">Dataset</div>
            <div class="sidebar-dataset-line">UCI · Obesity Levels</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_pills(labels: List[str]) -> None:
    html = "".join(f'<span class="source-pill">{label}</span>' for label in labels)
    st.markdown(html, unsafe_allow_html=True)


def get_active_theme() -> Dict[str, str]:
    """Return display-only colours matching Streamlit's active theme."""
    try:
        theme_type = str(st.context.theme.type).lower()
    except Exception:
        theme_type = "light"

    if theme_type == "dark":
        # Presentation only: chart data, labels, ordering and calculations are unchanged.
        return {
            "template": "none",
            "text": "#F5F5F7",
            "grid": "rgba(255,255,255,0.08)",
            "line": "rgba(255,255,255,0.12)",
            "hover_bg": "#2C2C2E",
            "hover_text": "#F5F5F7",
            "marker_contrast": "#F5F5F7",
        }

    return {
        "template": "plotly_white",
        "text": "#31464F",
        "grid": "#E8EFF0",
        "line": "#D7E3E7",
        "hover_bg": "#163B4D",
        "hover_text": "#FFFFFF",
        "marker_contrast": "#16303A",
    }


def base_plot_layout(fig: go.Figure, height: int = 470) -> go.Figure:
    """
    Apply presentation-only Plotly theming.
    Chart data, calculations, labels and class ordering are unchanged.
    """
    theme = get_active_theme()

    fig.update_layout(
        height=height,
        template=theme["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=70, b=30),
        font=dict(
            family="Arial",
            size=13,
            color=theme["text"],
        ),
        title_font=dict(
            size=18,
            color=theme["text"],
        ),
        legend=dict(
            font=dict(color=theme["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor=theme["hover_bg"],
            bordercolor=theme["line"],
            font=dict(
                size=12,
                color=theme["hover_text"],
            ),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=theme["grid"],
        zeroline=False,
        linecolor=theme["line"],
        tickfont=dict(color=theme["text"]),
        title_font=dict(color=theme["text"]),
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=theme["line"],
        tickfont=dict(color=theme["text"]),
        title_font=dict(color=theme["text"]),
    )

    return fig




PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "displayModeBar": True,
    "toImageButtonOptions": {
        "format": "png",
        "scale": 2,
    },
}


def render_plotly(
    fig: go.Figure,
    height: Optional[int] = None,
    key: Optional[str] = None,
) -> None:
    """Render every interactive figure through one responsive configuration."""
    resolved_height = int(height or fig.layout.height or 500)
    fig.update_layout(
        height=resolved_height,
        autosize=True,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        height=resolved_height,
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )


# =============================================================================
# DATA LOADING / PREPARATION FOR DASHBOARD ANALYTICS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    path = first_existing_path([
        "ObesityDataSet.csv",
        "data/ObesityDataSet.csv",
    ])
    if path is None:
        raise FileNotFoundError(
            "The project dataset is unavailable."
        )

    raw = pd.read_csv(path)
    clean = raw.drop_duplicates().copy()

    # EDA-only derived variables. They are NOT fed into the trained models.
    clean["BMI"] = clean["Weight"] / (clean["Height"] ** 2)
    clean["AgeGroup"] = pd.cut(
        clean["Age"],
        bins=[-np.inf, 19, 30, 45, np.inf],
        labels=[
            "Teenager (≤19)",
            "Young Adult (20–30)",
            "Adult (31–45)",
            "Senior (46+)",
        ],
        include_lowest=True,
    )
    return raw, clean, path


@st.cache_data(show_spinner=False)
def load_joblib_data(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_first_joblib(candidates: Tuple[str, ...]):
    """Load the first available saved evaluation object from candidate paths."""
    path = first_existing_path(list(candidates))
    if path is None:
        raise FileNotFoundError("Required evaluation data is unavailable.")
    return joblib.load(path), path


@st.cache_resource(show_spinner=False)
def load_prediction_model(model_name: str):
    """
    Load a trained classifier for live prediction.

    Prediction availability is intentionally independent from the saved
    evaluation results. A model only needs to load successfully and accept the
    established 23-feature input schema.
    """
    errors = []
    for model_path in MODEL_REGISTRY[model_name]["model_candidates"]:
        if not Path(model_path).exists():
            continue
        try:
            model = load_model(model_path)
            return model, model_path
        except Exception as exc:
            errors.append(f"{model_path}: {exc}")

    if errors:
        raise RuntimeError(f"{model_name} could not be loaded for prediction.")
    raise FileNotFoundError(f"{model_name} trained model is unavailable.")


def get_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    try:
        return load_dataset()
    except Exception as exc:
        st.error(f"Dataset unavailable: {exc}")
        return None, None, None


# =============================================================================
# STATISTICAL / ASSOCIATION HELPERS
# =============================================================================

def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    if frame.empty:
        return np.nan
    grand_mean = frame["value"].mean()
    ss_total = ((frame["value"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group_df in frame.groupby("group", observed=True):
        ss_between += len(group_df) * (group_df["value"].mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0 or min(observed.shape) < 2:
        return 0.0

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    valid = expected > 0
    chi2 = np.sum(((observed - expected) ** 2 / np.where(valid, expected, 1))[valid])
    phi2 = chi2 / n

    r, k = observed.shape
    # Bias-corrected Cramér's V.
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def iqr_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERICAL_COLUMNS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append(
            {
                "Feature": col,
                "IQR Lower Bound": lower,
                "IQR Upper Bound": upper,
                "Flagged Records": int(mask.sum()),
                "Flagged %": float(mask.mean()),
                "Observed Min": float(df[col].min()),
                "Observed Max": float(df[col].max()),
            }
        )
    return pd.DataFrame(rows)


def distribution_shape(skew: float) -> str:
    if skew >= 1.0:
        return "strongly right-skewed"
    if skew >= 0.5:
        return "moderately right-skewed"
    if skew > 0.1:
        return "slightly right-skewed"
    if skew <= -1.0:
        return "strongly left-skewed"
    if skew <= -0.5:
        return "moderately left-skewed"
    if skew < -0.1:
        return "slightly left-skewed"
    return "approximately symmetric"


def numeric_distribution_insight(df: pd.DataFrame, feature: str) -> str:
    s = df[feature].dropna()
    skew = float(s.skew())
    mean = float(s.mean())
    median = float(s.median())
    eta = eta_squared(df[feature], df[TARGET])
    unit = NUMERIC_UNITS[feature]
    return (
        f"{feature} is {distribution_shape(skew)} (skewness {skew:.2f}). "
        f"Its mean is {mean:.2f} and median is {median:.2f} {unit}. "
        f"As a descriptive class-separation measure, obesity-category grouping accounts for "
        f"approximately {eta:.1%} of the observed variance in {feature}. This is association, "
        "not evidence that the feature causes an obesity category."
    )


def class_trend_insight(df: pd.DataFrame, feature: str) -> str:
    means = df.groupby(TARGET, observed=True)[feature].mean().reindex(OBESITY_ORDER)
    first = float(means.iloc[0])
    last = float(means.iloc[-1])
    diffs = np.diff(means.to_numpy())
    reversals = int(np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))) if len(diffs) > 1 else 0
    max_class = means.idxmax()
    min_class = means.idxmin()

    if last > first:
        direction = "higher overall"
    elif last < first:
        direction = "lower overall"
    else:
        direction = "similar overall"

    if reversals == 0:
        shape_text = "The class means follow a broadly monotonic pattern."
    else:
        shape_text = (
            f"The pattern is not perfectly monotonic; there are {reversals} direction changes "
            "across neighbouring class means."
        )

    return (
        f"The mean {feature} is {direction} in {display_label(OBESITY_ORDER[-1])} "
        f"({last:.2f}) than in {display_label(OBESITY_ORDER[0])} ({first:.2f}). "
        f"The highest class mean occurs in {display_label(max_class)} and the lowest in "
        f"{display_label(min_class)}. {shape_text} Overlap between box distributions means this "
        "feature should be interpreted as one component of a multivariable classification problem."
    )


def target_association_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows = []
    for feature in NUMERICAL_COLUMNS:
        numeric_rows.append(
            {"Numerical Feature": feature, "Eta-squared": eta_squared(df[feature], df[TARGET])}
        )
    numeric_df = pd.DataFrame(numeric_rows).sort_values("Eta-squared", ascending=False)

    cat_rows = []
    for feature in CATEGORICAL_COLUMNS:
        cat_rows.append(
            {"Categorical Feature": feature, "Cramer's V": cramers_v(df[feature], df[TARGET])}
        )
    cat_df = pd.DataFrame(cat_rows).sort_values("Cramer's V", ascending=False)
    return numeric_df, cat_df


# =============================================================================
# DATA VISUALISATIONS
# =============================================================================

def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET].value_counts().reindex(OBESITY_ORDER, fill_value=0)
    chart_df = pd.DataFrame(
        {
            "Class": OBESITY_ORDER,
            "Label": [display_label(c) for c in OBESITY_ORDER],
            "Count": counts.values,
            "Percentage": counts.values / len(df),
            "Color": [OBESITY_COLORS[c] for c in OBESITY_ORDER],
        }
    )

    fig = go.Figure(
        go.Bar(
            x=chart_df["Count"],
            y=chart_df["Label"],
            orientation="h",
            marker_color=chart_df["Color"],
            customdata=np.column_stack([chart_df["Percentage"]]),
            text=[f"{c:,}  ·  {p:.1%}" for c, p in zip(chart_df["Count"], chart_df["Percentage"])],
            textposition="outside",
            hovertemplate="%{y}<br>Count: %{x:,}<br>Share: %{customdata[0]:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Obesity Class Distribution — Cleaned Modelling Dataset",
        xaxis_title="Number of observations",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    max_count = max(chart_df["Count"].max(), 1)
    fig.update_xaxes(range=[0, max_count * 1.20])
    return base_plot_layout(fig, 460)


def plot_numeric_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    series = df[feature].dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=32,
            marker_color="#3A8688",
            opacity=0.82,
            name="Observations",
            hovertemplate=f"{feature}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        )
    )
    mean = series.mean()
    median = series.median()
    fig.add_vline(
        x=mean,
        line_width=2,
        line_dash="dash",
        line_color="#1F5F75",
        annotation_text=f"Mean {mean:.2f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median,
        line_width=2,
        line_dash="dot",
        line_color="#6B7F93",
        annotation_text=f"Median {median:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Distribution of {feature}",
        xaxis_title=f"{feature} — {NUMERIC_UNITS[feature]}",
        yaxis_title="Number of observations",
        bargap=0.04,
        showlegend=False,
    )
    return base_plot_layout(fig, 450)


def numeric_question(feature: str) -> str:
    questions = {
        "Age": "Are age distributions different across obesity categories?",
        "Weight": "How does weight change across obesity categories?",
        "Height": "Does height differ meaningfully across obesity categories?",
        "FCVC": "How does vegetable-consumption frequency vary across obesity categories?",
        "NCP": "Do main-meal patterns differ across obesity categories?",
        "CH2O": "How does reported water consumption vary across obesity categories?",
        "FAF": "Does physical-activity frequency decline across higher obesity categories?",
        "TUE": "Does technology-use frequency differ across obesity categories?",
    }
    return questions.get(feature, f"How does {feature} vary across obesity categories?")


def plot_numeric_by_class(df: pd.DataFrame, feature: str) -> go.Figure:
    chart_df = df[[TARGET, feature]].copy()
    chart_df["Obesity Category"] = chart_df[TARGET].map(DISPLAY_LABELS)

    hover_formats = {
        "Age": (".2f", " years"),
        "Height": (".2f", " m"),
        "Weight": (".2f", " kg"),
        "FCVC": (".2f", ""),
        "NCP": (".2f", ""),
        "CH2O": (".2f", ""),
        "FAF": (".2f", ""),
        "TUE": (".2f", ""),
    }
    value_format, unit_suffix = hover_formats.get(feature, (".2f", ""))

    fig = go.Figure()
    for cls in OBESITY_ORDER:
        subset = chart_df[chart_df[TARGET] == cls]
        readable_class = display_label(cls)
        fig.add_trace(
            go.Box(
                x=subset[feature],
                y=[readable_class] * len(subset),
                name=readable_class,
                marker_color=OBESITY_COLORS[cls],
                line_color=OBESITY_COLORS[cls],
                boxpoints="outliers",
                orientation="h",
                showlegend=False,
                hoveron="boxes+points",
                hovertemplate=(
                    f"<b>{readable_class}</b><br>"
                    f"Feature: {feature}<br>"
                    f"Value: %{{x:{value_format}}}{unit_suffix}"
                    "<extra></extra>"
                ),
            )
        )

    means = df.groupby(TARGET, observed=True)[feature].mean().reindex(OBESITY_ORDER)
    fig.add_trace(
        go.Scatter(
            x=means.values,
            y=[display_label(c) for c in OBESITY_ORDER],
            mode="markers",
            name="Class mean",
            marker=dict(
                symbol="diamond",
                size=9,
                color=get_active_theme()["marker_contrast"],
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Mean {feature}: %{{x:{value_format}}}{unit_suffix}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=numeric_question(feature),
        xaxis_title=f"{feature} — {NUMERIC_UNITS[feature]}",
        yaxis_title="",
        hovermode="closest",
        hoverdistance=20,
        spikedistance=-1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[display_label(c) for c in OBESITY_ORDER],
        autorange="reversed",
    )
    return base_plot_layout(fig, 560)

def plot_weight_height(df: pd.DataFrame) -> go.Figure:
    chart_df = df.copy()
    chart_df["Obesity Category"] = chart_df[TARGET].map(DISPLAY_LABELS)

    fig = px.scatter(
        chart_df,
        x="Height",
        y="Weight",
        color=TARGET,
        category_orders={TARGET: OBESITY_ORDER},
        color_discrete_map=OBESITY_COLORS,
        hover_data={
            TARGET: False,
            "Obesity Category": True,
            "Age": ":.1f",
            "Gender": True,
            "Height": ":.3f",
            "Weight": ":.1f",
            "BMI": ":.2f",
        },
        opacity=0.68,
        render_mode="webgl",
        labels={TARGET: "Obesity category"},
        title="Weight vs Height by Obesity Category",
    )
    fig.update_layout(
        xaxis_title="Height (m)",
        yaxis_title="Weight (kg)",
        legend_title_text="Obesity category",
    )
    return base_plot_layout(fig, 560)


def plot_categorical_relationship(df: pd.DataFrame, feature: str) -> go.Figure:
    counts = (
        df.groupby([feature, TARGET], observed=True)
        .size()
        .reset_index(name="Count")
    )
    totals = counts.groupby(feature)["Count"].transform("sum")
    counts["Percentage"] = counts["Count"] / totals
    counts["Feature Label"] = counts[feature].map(humanize_category)

    observed_order = [v for v in EXPECTED_CATEGORIES.get(feature, []) if v in df[feature].unique()]
    if not observed_order:
        observed_order = list(df[feature].dropna().unique())
    x_order = [humanize_category(v) for v in observed_order]

    fig = go.Figure()
    for cls in OBESITY_ORDER:
        subset = counts[counts[TARGET] == cls].copy()
        # Reindex to retain factor-category order even if a class/category combination is absent.
        lookup = subset.set_index(feature)
        percentages = []
        raw_counts = []
        for value in observed_order:
            if value in lookup.index:
                row = lookup.loc[value]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                percentages.append(float(row["Percentage"]))
                raw_counts.append(int(row["Count"]))
            else:
                percentages.append(0.0)
                raw_counts.append(0)

        fig.add_trace(
            go.Bar(
                x=x_order,
                y=percentages,
                name=display_label(cls),
                marker_color=OBESITY_COLORS[cls],
                customdata=np.column_stack([raw_counts]),
                text=[f"{p:.0%}" if p >= 0.08 else "" for p in percentages],
                textposition="inside",
                hovertemplate=(
                    "%{x}<br>" + display_label(cls) +
                    "<br>Share within group: %{y:.2%}<br>Count: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Obesity Composition Within {humanize_category(feature)} Groups",
        xaxis_title=humanize_category(feature),
        yaxis_title="Percentage within selected group",
        barmode="stack",
        legend_title_text="Obesity category",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return base_plot_layout(fig, 520)


def categorical_insight(df: pd.DataFrame, feature: str) -> str:
    severe = df[TARGET].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])
    summary = (
        df.assign(_severe=severe)
        .groupby(feature, observed=True)
        .agg(Group_Size=(TARGET, "size"), Severe_Share=("_severe", "mean"))
        .sort_values("Severe_Share", ascending=False)
    )
    highest = summary.iloc[0]
    lowest = summary.iloc[-1]
    high_name = humanize_category(summary.index[0])
    low_name = humanize_category(summary.index[-1])
    v = cramers_v(df[feature], df[TARGET])

    small_group_note = ""
    if int(highest["Group_Size"]) < 30 or int(lowest["Group_Size"]) < 30:
        small_group_note = (
            " At least one compared group has fewer than 30 observations, so the percentage "
            "difference should be interpreted cautiously."
        )

    return (
        f"The strongest severe-obesity share within {humanize_category(feature)} occurs for "
        f"**{high_name}** ({highest['Severe_Share']:.1%}, n={int(highest['Group_Size'])}), "
        f"compared with **{low_name}** ({lowest['Severe_Share']:.1%}, n={int(lowest['Group_Size'])}). "
        f"Bias-corrected Cramér's V between {feature} and obesity category is **{v:.3f}**. "
        "This is a descriptive association and should not be interpreted as causation."
        + small_group_note
    )


def plot_correlation_heatmap(df: pd.DataFrame, method: str) -> go.Figure:
    corr = df[NUMERICAL_COLUMNS].corr(method=method.lower())
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, "#3B6B82"],
                [0.5, "#F7FAFA"],
                [1.0, "#2B7A78"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar_title=f"{method} r",
        )
    )
    fig.update_layout(
        title=f"{method} Correlation Matrix — Numerical Predictors Only",
        xaxis_title="",
        yaxis_title="",
    )
    return base_plot_layout(fig, 620)


# =============================================================================
# MODEL EVALUATION HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def build_project_evaluation_data() -> Dict[str, object]:
    """
    Reproduce the established modelling partition exactly.

    Important: this intentionally mirrors the project's existing categorical
    encoding behaviour. It does not create a new split or alter preprocessing.
    """
    raw, _, _ = load_dataset()
    modelling_df = raw.drop_duplicates().copy()

    X = modelling_df.drop(columns=[TARGET])
    y = modelling_df[[TARGET]]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_train_encoded = pd.get_dummies(
        X_train,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )
    X_test_encoded = pd.get_dummies(
        X_test,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )

    # This is the exact alignment rule used by the established modelling run.
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )

    X_train_encoded = X_train_encoded.astype(float)
    X_test_encoded = X_test_encoded.astype(float)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_encoded": X_train_encoded,
        "X_test_encoded": X_test_encoded,
        "y_train": np.asarray(y_train).ravel().astype(str),
        "y_test": np.asarray(y_test).ravel().astype(str),
    }


def get_model_classes(model_name: str, model) -> np.ndarray:
    if model_name == "XGBoost":
        return XGB_CLASS_NAMES.copy()

    if hasattr(model, "classes_"):
        return np.asarray(model.classes_).astype(str)

    if hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        if hasattr(final_estimator, "classes_"):
            return np.asarray(final_estimator.classes_).astype(str)

    raise AttributeError(f"Could not determine class order for {model_name}.")


def get_model_feature_names(model_name: str, model, X_test: Optional[pd.DataFrame]) -> List[str]:
    if model_name == "XGBoost" and hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names:
            return [str(x) for x in names]

    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if X_test is not None and hasattr(X_test, "columns"):
        return [str(x) for x in X_test.columns]

    raise AttributeError(f"Feature names are unavailable for {model_name}.")


def calculate_multiclass_roc_auc(
    y_true,
    y_proba,
    class_names,
    average="weighted",
) -> float:
    """Calculate multiclass One-vs-Rest ROC-AUC in probability-column order."""
    y_true = np.asarray(y_true).ravel().astype(str)
    y_proba = np.asarray(y_proba, dtype=float)
    class_names = np.asarray(class_names).ravel().astype(str)

    if y_proba.ndim > 2:
        y_proba = np.squeeze(y_proba)
    if y_proba.ndim != 2:
        raise ValueError("Probability output must be a 2-D array.")
    if len(y_true) != y_proba.shape[0]:
        raise ValueError("Probability row count does not match the held-out labels.")
    if y_proba.shape[1] != len(class_names):
        raise ValueError("Probability-column count does not match the model class count.")
    if len(np.unique(class_names)) != len(class_names):
        raise ValueError("Model class order contains duplicate labels.")
    if not set(np.unique(y_true)).issubset(set(class_names)):
        raise ValueError("Held-out labels fall outside the model class order.")
    if not np.isfinite(y_proba).all():
        raise ValueError("Probability output contains non-finite values.")
    if np.any(y_proba < -1e-9) or np.any(y_proba > 1.0 + 1e-9):
        raise ValueError("Probability output contains values outside [0, 1].")
    if not np.allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("Probability rows do not sum to 1 within tolerance.")

    y_bin = label_binarize(y_true, classes=class_names)
    return float(
        roc_auc_score(
            y_bin,
            y_proba,
            average=average,
            multi_class="ovr",
        )
    )


def calculate_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel().astype(str)
    y_pred = np.asarray(y_pred).ravel().astype(str)
    errors = int(np.sum(y_true != y_pred))
    test_size = int(len(y_true))

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC-AUC": calculate_multiclass_roc_auc(
            y_true,
            y_proba,
            classes,
            average="weighted",
        ),
        "Errors": errors,
        "Error Rate": float(errors / test_size) if test_size else np.nan,
        "Test Size": test_size,
    }


def _decode_predictions(model_name: str, raw_predictions: np.ndarray) -> np.ndarray:
    raw_predictions = np.asarray(raw_predictions).ravel()
    if model_name != "XGBoost":
        return raw_predictions.astype(str)

    raw_as_str = raw_predictions.astype(str)
    if set(np.unique(raw_as_str)).issubset(set(XGB_CLASS_NAMES)):
        return raw_as_str

    ids = raw_predictions.astype(int)
    if np.any(ids < 0) or np.any(ids >= len(XGB_CLASS_NAMES)):
        raise ValueError("XGBoost returned an unknown class ID.")
    return XGB_CLASS_NAMES[ids].astype(str)


def _validate_feature_schema(model_name: str, model, X_eval: pd.DataFrame) -> None:
    expected_columns = [str(x) for x in X_eval.columns]

    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(expected_columns):
        raise ValueError("Model feature count does not match the 23-feature evaluation schema.")

    model_names = None
    if model_name == "XGBoost" and hasattr(model, "get_booster"):
        model_names = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        model_names = list(model.feature_names_in_)

    if model_names is not None and [str(x) for x in model_names] != expected_columns:
        raise ValueError("Model feature names/order do not match the evaluation schema.")


def _metrics_match_project(model_name: str, metrics: Dict[str, float]) -> bool:
    reference = PROJECT_TEST_RESULTS[model_name]
    for key in ["Accuracy", "Precision", "Recall", "Weighted F1", "Macro F1", "ROC-AUC"]:
        if abs(float(metrics[key]) - float(reference[key])) > MODEL_METRIC_TOLERANCE:
            return False
    return int(metrics["Errors"]) == int(reference["Errors"])


@st.cache_resource(show_spinner=False)
def load_evaluation_bundle(model_name: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load the project's saved held-out evaluation bundle.

    Evaluation and live prediction are deliberately separated:
    - evaluation uses the matching saved X_test / y_test / y_pred / y_proba set;
    - prediction loads the trained classifier independently.

    This prevents a model file from a different export generation from
    disabling the presentation's saved evaluation results.
    """
    config = MODEL_REGISTRY[model_name]
    eval_cfg = config.get("evaluation", {})

    try:
        X_test, _ = load_first_joblib(tuple(eval_cfg["X_test"]))
        y_test, _ = load_first_joblib(tuple(eval_cfg["y_test"]))
        y_pred, _ = load_first_joblib(tuple(eval_cfg["y_pred"]))
        y_proba, _ = load_first_joblib(tuple(eval_cfg["y_proba"]))
    except Exception:
        return None, f"{model_name} evaluation results are unavailable."

    try:
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        y_true = np.asarray(y_test).ravel().astype(str)
        y_pred = np.asarray(y_pred).ravel().astype(str)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim > 2:
            y_proba = np.squeeze(y_proba)

        if len(y_true) != len(y_pred):
            raise ValueError("Prediction length does not match the test labels.")
        if y_proba.ndim != 2 or y_proba.shape[0] != len(y_true):
            raise ValueError("Probability output does not match the test labels.")
        if y_proba.shape[1] != len(XGB_CLASS_NAMES):
            raise ValueError("Probability output must contain seven class columns.")
        if not np.allclose(y_proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("Probability rows do not sum to one.")

        # All four saved evaluation exports use the same seven-label ordering.
        # This is also the LabelEncoder.classes_ order used for XGBoost.
        classes = XGB_CLASS_NAMES.copy()
        if not set(np.unique(y_true)).issubset(set(classes)):
            raise ValueError("Unknown class found in the held-out labels.")
        if not set(np.unique(y_pred)).issubset(set(classes)):
            raise ValueError("Unknown class found in the saved predictions.")

        metrics = calculate_evaluation_metrics(
            y_true,
            y_pred,
            y_proba,
            classes,
        )

        # A trained model is optional for evaluation-only views such as the
        # confusion matrix and ROC curves. When available, attach it so the
        # feature-analysis tab can still use model-specific information.
        model = None
        model_path = None
        try:
            model, model_path = load_prediction_model(model_name)
        except Exception:
            pass

        return {
            "model": model,
            "model_path": model_path,
            "X_test": X_test,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classes": classes,
            "metrics": metrics,
        }, None

    except Exception:
        return None, f"{model_name} evaluation results are unavailable."


def get_cv_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Model": model_name, **CV_RESULTS[model_name]} for model_name in MODEL_REGISTRY]
    )


def build_comparison_dataframe() -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, str]]:
    rows: List[Dict] = []
    bundles: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}

    for model_name in MODEL_REGISTRY:
        bundle, error = load_evaluation_bundle(model_name)
        if bundle is None:
            errors[model_name] = error or f"{model_name} is unavailable."
            continue

        bundles[model_name] = bundle
        rows.append({"Model": model_name, **bundle["metrics"]})

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.merge(get_cv_dataframe(), on="Model", how="left", validate="one_to_one")

    return comparison, bundles, errors


def confusion_details(artifact: Dict) -> Dict:
    cm = confusion_matrix(
        artifact["y_true"],
        artifact["y_pred"],
        labels=OBESITY_ORDER,
    )
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0) * 100
    recalls = np.diag(cm_pct)

    best_idx = int(np.argmax(recalls))
    hardest_idx = int(np.argmin(recalls))

    best_pair = None
    best_pair_count = -1
    for i in range(len(OBESITY_ORDER)):
        for j in range(i + 1, len(OBESITY_ORDER)):
            total_pair = int(cm[i, j] + cm[j, i])
            if total_pair > best_pair_count:
                best_pair_count = total_pair
                best_pair = (OBESITY_ORDER[i], OBESITY_ORDER[j])

    return {
        "cm": cm,
        "cm_pct": cm_pct,
        "best_class": OBESITY_ORDER[best_idx],
        "best_recall": recalls[best_idx] / 100,
        "hardest_class": OBESITY_ORDER[hardest_idx],
        "hardest_recall": recalls[hardest_idx] / 100,
        "pair": best_pair,
        "pair_count": best_pair_count,
        "pair_adjacent": (
            abs(OBESITY_ORDER.index(best_pair[0]) - OBESITY_ORDER.index(best_pair[1])) == 1
            if best_pair else False
        ),
    }


def plot_confusion_matrix(artifact: Dict, mode: str) -> go.Figure:
    details = confusion_details(artifact)
    labels = [display_label(c) for c in OBESITY_ORDER]

    if mode == "Percentage":
        z = details["cm_pct"]
        text = [[f"{v:.1f}%" for v in row] for row in z]
        hover = "Actual: %{y}<br>Predicted: %{x}<br>Row share: %{z:.2f}%<extra></extra>"
        title = "Confusion Matrix — Row-wise Percentage"
        colorbar_title = "% of actual class"
    else:
        z = details["cm"]
        text = [[str(int(v)) for v in row] for row in z]
        hover = "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        title = "Confusion Matrix — Counts"
        colorbar_title = "Count"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#F7FAFA"], [1, "#25666D"]],
            text=text,
            texttemplate="%{text}",
            hovertemplate=hover,
            colorbar_title=colorbar_title,
        )
    )
    fig.update_layout(title=title, xaxis_title="Predicted category", yaxis_title="Actual category")
    fig.update_yaxes(autorange="reversed")
    return base_plot_layout(fig, 620)


def class_performance_dataframe(artifact: Dict) -> pd.DataFrame:
    report = classification_report(
        artifact["y_true"],
        artifact["y_pred"],
        labels=OBESITY_ORDER,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in OBESITY_ORDER:
        values = report.get(cls, {})
        rows.append(
            {
                "Class": cls,
                "Obesity Category": display_label(cls),
                "Precision": values.get("precision", np.nan),
                "Recall": values.get("recall", np.nan),
                "F1": values.get("f1-score", np.nan),
                "Support": int(values.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def plot_class_performance(class_df: pd.DataFrame) -> go.Figure:
    long = class_df.melt(
        id_vars=["Class", "Obesity Category"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        long,
        y="Obesity Category",
        x="Score",
        color="Metric",
        barmode="group",
        orientation="h",
        color_discrete_map={"Precision": "#6B7F93", "Recall": "#3A8688", "F1": "#1F5F75"},
        title="Class-level Precision, Recall and F1",
        category_orders={"Obesity Category": [display_label(c) for c in OBESITY_ORDER]},
        text_auto=".2f",
    )
    fig.update_layout(xaxis_title="Score", yaxis_title="")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return base_plot_layout(fig, 600)


def compute_roc_details(artifact: Dict) -> Dict:
    y_true = artifact["y_true"]
    y_proba = artifact["y_proba"]
    classes = np.asarray(artifact["classes"]).astype(str)

    fpr = {}
    tpr = {}
    class_auc = {}

    for cls in OBESITY_ORDER:
        idx_matches = np.where(classes == cls)[0]
        if len(idx_matches) != 1:
            continue
        idx = int(idx_matches[0])
        binary_true = (y_true == cls).astype(int)
        cls_fpr, cls_tpr, _ = roc_curve(binary_true, y_proba[:, idx])

        # Defensive endpoint handling for presentation consistency.
        if cls_fpr[0] != 0.0 or cls_tpr[0] != 0.0:
            cls_fpr = np.insert(cls_fpr, 0, 0.0)
            cls_tpr = np.insert(cls_tpr, 0, 0.0)
        if cls_fpr[-1] != 1.0 or cls_tpr[-1] != 1.0:
            cls_fpr = np.append(cls_fpr, 1.0)
            cls_tpr = np.append(cls_tpr, 1.0)

        fpr[cls] = cls_fpr
        tpr[cls] = cls_tpr
        class_auc[cls] = float(auc(cls_fpr, cls_tpr))

    y_bin = label_binarize(y_true, classes=classes)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = float(auc(fpr_micro, tpr_micro))

    all_fpr = np.unique(np.concatenate([fpr[c] for c in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in fpr:
        mean_tpr += np.interp(all_fpr, fpr[cls], tpr[cls])
    mean_tpr /= max(len(fpr), 1)
    auc_macro_curve = float(auc(all_fpr, mean_tpr))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "class_auc": class_auc,
        "micro": (fpr_micro, tpr_micro, auc_micro),
        "macro": (all_fpr, mean_tpr, auc_macro_curve),
    }


def plot_roc_classes(roc_details: Dict) -> go.Figure:
    fig = go.Figure()

    for cls in OBESITY_ORDER:
        if cls not in roc_details["fpr"]:
            continue

        class_auc = roc_details["class_auc"][cls]
        readable = display_label(cls)

        fig.add_trace(
            go.Scatter(
                x=roc_details["fpr"][cls],
                y=roc_details["tpr"][cls],
                mode="lines",
                name=f"{readable} · AUC {class_auc:.3f}",
                line=dict(color=OBESITY_COLORS[cls], width=2),
                hovertemplate=(
                    f"<b>{readable}</b><br>"
                    "False Positive Rate: %{x:.3f}<br>"
                    "True Positive Rate: %{y:.3f}<br>"
                    f"AUC: {class_auc:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline · AUC 0.500",
            line=dict(color="#7A8B92", dash="dash", width=1.5),
            hovertemplate=(
                "<b>Random baseline</b><br>"
                "False Positive Rate: %{x:.3f}<br>"
                "True Positive Rate: %{y:.3f}<br>"
                "AUC: 0.500<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="One-vs-Rest ROC Curves by Obesity Category",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
            font=dict(size=11),
            entrywidth=205,
            entrywidthmode="pixels",
        ),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    fig = base_plot_layout(fig, 620)
    fig.update_layout(margin=dict(l=24, r=24, t=68, b=155))
    return fig

def plot_roc_averages(roc_details: Dict) -> go.Figure:
    micro_fpr, micro_tpr, micro_auc = roc_details["micro"]
    macro_fpr, macro_tpr, macro_auc = roc_details["macro"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=micro_fpr,
            y=micro_tpr,
            mode="lines",
            name=f"Micro-average · AUC {micro_auc:.3f}",
            line=dict(color="#1F5F75", width=3),
            hovertemplate=(
                "<b>Micro-average</b><br>"
                "False Positive Rate: %{x:.3f}<br>"
                "True Positive Rate: %{y:.3f}<br>"
                f"AUC: {micro_auc:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_fpr,
            y=macro_tpr,
            mode="lines",
            name=f"Macro-average · AUC {macro_auc:.3f}",
            line=dict(color="#3A8688", width=3, dash="dot"),
            hovertemplate=(
                "<b>Macro-average</b><br>"
                "False Positive Rate: %{x:.3f}<br>"
                "True Positive Rate: %{y:.3f}<br>"
                f"AUC: {macro_auc:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline · AUC 0.500",
            line=dict(color="#7A8B92", dash="dash", width=1.5),
            hovertemplate=(
                "<b>Random baseline</b><br>"
                "False Positive Rate: %{x:.3f}<br>"
                "True Positive Rate: %{y:.3f}<br>"
                "AUC: 0.500<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Micro- and Macro-average ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig = base_plot_layout(fig, 540)
    fig.update_layout(margin=dict(l=24, r=24, t=68, b=105))
    return fig

def strip_transformer_prefix(name: str) -> str:
    return str(name).split("__", 1)[-1]


def model_feature_analysis(model_name: str, artifact: Dict) -> Tuple[Optional[pd.DataFrame], str, str]:
    model = artifact.get("model")
    X_test = artifact["X_test"]

    if model is None and model_name != "K-Nearest Neighbours (KNN)":
        path = first_existing_path(MODEL_REGISTRY[model_name].get("feature_data_candidates", []))
        if path is not None:
            try:
                df = pd.read_csv(path)
                importance_col = next(
                    (c for c in ["Importance", "Mean Importance", "Importance (mean drop in accuracy)"] if c in df.columns),
                    None,
                )
                if "Feature" in df.columns and importance_col is not None:
                    out = df[["Feature", importance_col]].copy()
                    out.columns = ["Feature", "Importance"]
                    return (
                        out.sort_values("Importance", ascending=False).head(15),
                        f"{model_name} feature analysis",
                        "Feature values summarise the prepared model analysis for this classifier.",
                    )
            except Exception:
                pass
        return None, "Feature analysis unavailable", "Feature analysis is unavailable for this classifier."

    if model_name == "K-Nearest Neighbours (KNN)":
        path = first_existing_path(MODEL_REGISTRY[model_name]["feature_data_candidates"])
        if path is None:
            return (
                None,
                "KNN has no intrinsic feature importance.",
                "Permutation importance is the appropriate post-hoc feature view for KNN. The prepared ranking is not available in this presentation.",
            )
        df = pd.read_csv(path)
        possible = [
            "Importance (mean drop in accuracy)",
            "Importance",
            "Mean Importance",
        ]
        importance_col = next((c for c in possible if c in df.columns), None)
        if "Feature" not in df.columns or importance_col is None:
            return None, "Permutation importance unavailable", "The prepared KNN feature ranking could not be read."
        out = df[["Feature", importance_col]].copy()
        out.columns = ["Feature", "Importance"]
        out = out.sort_values("Importance", ascending=False).head(15)
        return (
            out,
            "KNN permutation importance",
            "Permutation importance measures the drop in predictive performance when a feature is shuffled; it is not built into KNN.",
        )

    if model_name == "Logistic Regression":
        estimator = model
        feature_names = None
        if hasattr(model, "named_steps"):
            final_estimator = model.named_steps.get("logistic_regression")
            if final_estimator is None:
                final_estimator = list(model.named_steps.values())[-1]
            estimator = final_estimator

            preprocessor = model.named_steps.get("preprocessor")
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                try:
                    input_names = get_model_feature_names(model_name, model, X_test)
                    feature_names = [
                        strip_transformer_prefix(x)
                        for x in preprocessor.get_feature_names_out(input_names)
                    ]
                except Exception:
                    feature_names = None

        if not hasattr(estimator, "coef_"):
            return None, "Coefficient analysis unavailable", "The fitted Logistic Regression coefficients could not be accessed."

        coefficients = np.asarray(estimator.coef_)
        importance = np.mean(np.abs(coefficients), axis=0)
        if feature_names is None:
            feature_names = get_model_feature_names(model_name, model, X_test)
        if len(feature_names) != len(importance):
            return None, "Coefficient analysis unavailable", "Coefficient count does not match available feature names."

        out = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        out = out.sort_values("Importance", ascending=False).head(15)
        return (
            out,
            "Logistic Regression mean absolute coefficient magnitude",
            "Coefficient magnitude indicates how strongly a standardised model coefficient contributes across the seven one-vs-rest/multiclass coefficient vectors. It is not tree feature importance and does not imply causality.",
        )

    if hasattr(model, "feature_importances_"):
        feature_names = get_model_feature_names(model_name, model, X_test)
        values = np.asarray(model.feature_importances_)
        if len(feature_names) != len(values):
            return None, "Tree feature importance unavailable", "Importance count does not match feature names."
        out = pd.DataFrame({"Feature": feature_names, "Importance": values})
        out = out.sort_values("Importance", ascending=False).head(15)
        if model_name == "XGBoost":
            title = "XGBoost gain-based feature importance"
            note = "The fitted XGBoost model uses tree-based importance_type = 'gain'. Importance is model association, not causal effect."
        else:
            title = "Random Forest tree-based feature importance"
            note = "Random Forest importance reflects impurity reduction across fitted trees. Importance is model association, not causal effect."
        return out, title, note

    return None, "Feature analysis unavailable", "A supported feature-analysis view is not available for this model."


def plot_feature_analysis(feature_df: pd.DataFrame, title: str) -> go.Figure:
    chart = feature_df.sort_values("Importance", ascending=True).copy()
    fig = go.Figure(
        go.Bar(
            x=chart["Importance"],
            y=chart["Feature"],
            orientation="h",
            marker_color="#2B7A78",
            text=[f"{v:.3f}" for v in chart["Importance"]],
            textposition="outside",
            hovertemplate="%{y}<br>Value: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Importance / coefficient magnitude", yaxis_title="")
    return base_plot_layout(fig, 560)


def original_feature_name(encoded_name: str) -> str:
    name = strip_transformer_prefix(encoded_name)
    if name in NUMERICAL_COLUMNS:
        return name
    for base in CATEGORICAL_COLUMNS:
        if name == base or name.startswith(base + "_"):
            return base
    return name


def aggregate_feature_importance(feature_df: pd.DataFrame) -> pd.DataFrame:
    work = feature_df.copy()
    work["Original Feature"] = work["Feature"].map(original_feature_name)
    return (
        work.groupby("Original Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

def get_prediction_columns(model_name: str, model) -> List[str]:
    project_columns = build_project_evaluation_data()["X_test_encoded"].columns.tolist()
    project_set = set(project_columns)

    model_names = None

    if model_name == "XGBoost" and hasattr(model, "get_booster"):
        booster_names = model.get_booster().feature_names
        if booster_names:
            model_names = [str(x) for x in booster_names]

    if model_names is None and hasattr(model, "feature_names_in_"):
        model_names = [str(x) for x in model.feature_names_in_]

    if model_names is not None:
        if len(model_names) != 23 or set(model_names) != project_set:
            raise ValueError("Model feature schema is incompatible with the 23 project predictors.")
        # Preserve the exact order expected by this trained model.
        return model_names

    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != 23:
        raise ValueError("Model feature count is incompatible with the project input schema.")

    return project_columns


def build_encoded_prediction_row(model_name: str, model, values: Dict) -> pd.DataFrame:
    columns = get_prediction_columns(model_name, model)
    row = pd.DataFrame(0.0, index=[0], columns=columns)

    for col in NUMERICAL_COLUMNS:
        if col in row.columns:
            row.at[0, col] = float(values[col])

    categorical_values = {
        "Gender": values["Gender"],
        "family_history_with_overweight": values["family_history_with_overweight"],
        "FAVC": values["FAVC"],
        "CAEC": values["CAEC"],
        "SMOKE": values["SMOKE"],
        "SCC": values["SCC"],
        "CALC": values["CALC"],
        "MTRANS": values["MTRANS"],
    }

    # Training used pd.get_dummies(..., drop_first=True). If a category's dummy
    # column is absent, that category is the training reference level and the
    # all-zero representation is therefore correct.
    for feature, selected_value in categorical_values.items():
        dummy_name = f"{feature}_{selected_value}"
        if dummy_name in row.columns:
            row.at[0, dummy_name] = 1.0

    return row


def make_prediction(model_name: str, input_values: Dict) -> Dict:
    model, _ = load_prediction_model(model_name)
    encoded_row = build_encoded_prediction_row(model_name, model, input_values)

    raw_pred = np.asarray(model.predict(encoded_row)).ravel()[0]
    proba = np.asarray(model.predict_proba(encoded_row)[0], dtype=float)

    if model_name == "XGBoost":
        if str(raw_pred) in set(XGB_CLASS_NAMES):
            pred_label = str(raw_pred)
        else:
            pred_label = str(XGB_CLASS_NAMES[int(raw_pred)])
        proba_classes = XGB_CLASS_NAMES.copy()
    else:
        pred_label = str(raw_pred)
        proba_classes = get_model_classes(model_name, model)

    if len(proba) != len(proba_classes):
        raise ValueError("Prediction probability count does not match model class count.")
    if not np.isclose(proba.sum(), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("Prediction probabilities do not sum to 1 within tolerance.")

    return {
        "model": model_name,
        "prediction": pred_label,
        "confidence": float(proba.max()),
        "classes": proba_classes,
        "probabilities": proba,
        "encoded_columns": list(encoded_row.columns),
    }

def cross_model_prediction_results(input_values: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the same submitted profile through each trained classifier without ensembling."""
    summary_rows = []
    probability_rows = []

    for model_name in MODEL_REGISTRY:
        try:
            result = make_prediction(model_name, input_values)
        except Exception:
            continue

        probability_map = {
            str(cls): float(prob)
            for cls, prob in zip(result["classes"], result["probabilities"])
        }
        summary_rows.append(
            {
                "Model": model_name,
                "Prediction": display_label(result["prediction"]),
                "Confidence": result["confidence"],
                "Raw Prediction": result["prediction"],
            }
        )
        row = {"Model": model_name}
        row.update({cls: probability_map.get(cls, 0.0) for cls in OBESITY_ORDER})
        probability_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(probability_rows)


def plot_cross_model_probability_heatmap(probability_df: pd.DataFrame) -> go.Figure:
    ordered = probability_df.set_index("Model").reindex(
        [name for name in MODEL_REGISTRY if name in probability_df["Model"].values]
    )
    z = ordered[OBESITY_ORDER].to_numpy(dtype=float)
    x_labels = [display_label(cls) for cls in OBESITY_ORDER]
    y_labels = [MODEL_SHORT_LABELS.get(name, name) for name in ordered.index]
    full_names = np.repeat(np.asarray(ordered.index, dtype=object)[:, None], len(OBESITY_ORDER), axis=1)

    theme = get_active_theme()
    if theme["template"] == "none":
        probability_scale = [
            [0.0, "#242426"],
            [0.35, "#2F4858"],
            [0.70, "#25666D"],
            [1.0, "#64D2FF"],
        ]
    else:
        probability_scale = [
            [0.0, "#F4F7F7"],
            [0.35, "#B8E0DF"],
            [0.70, "#3A8688"],
            [1.0, "#164A57"],
        ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=0,
            zmax=1,
            colorscale=probability_scale,
            text=np.vectorize(lambda value: f"{value:.0%}")(z),
            texttemplate="%{text}",
            customdata=full_names,
            hovertemplate=(
                "Model: %{customdata}<br>"
                "Class: %{x}<br>"
                "Probability: %{z:.2%}<extra></extra>"
            ),
            colorbar=dict(title="Probability", tickformat=".0%"),
        )
    )
    fig.update_layout(
        title="Cross-Model Probability Comparison",
        xaxis_title="Obesity category",
        yaxis_title="Model",
    )
    fig.update_xaxes(tickangle=-18)
    return base_plot_layout(fig, 440)


def profile_percentile_dataframe(clean: pd.DataFrame, input_values: Dict) -> pd.DataFrame:
    rows = []
    for feature in NUMERICAL_COLUMNS:
        series = clean[feature].dropna().astype(float)
        selected = float(input_values[feature])
        percentile = float((series <= selected).mean())
        rows.append(
            {
                "Feature": feature,
                "Selected Value": selected,
                "Dataset Median": float(series.median()),
                "Percentile": percentile,
                "Unit": NUMERIC_UNITS[feature],
            }
        )
    return pd.DataFrame(rows)


def plot_profile_percentiles(profile_df: pd.DataFrame) -> go.Figure:
    chart = profile_df.sort_values("Percentile", ascending=True).copy()
    customdata = np.empty((len(chart), 3), dtype=object)
    customdata[:, 0] = chart["Selected Value"].to_numpy(dtype=float)
    customdata[:, 1] = chart["Dataset Median"].to_numpy(dtype=float)
    customdata[:, 2] = chart["Unit"].astype(str).to_numpy()
    fig = go.Figure(
        go.Bar(
            x=chart["Percentile"],
            y=chart["Feature"],
            orientation="h",
            marker_color="#3A8688",
            text=[f"{value:.0%}" for value in chart["Percentile"]],
            textposition="outside",
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Dataset percentile: %{x:.1%}<br>"
                "Selected value: %{customdata[0]:.2f} %{customdata[2]}<br>"
                "Dataset median: %{customdata[1]:.2f} %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=0.5,
        line_dash="dot",
        line_width=1.5,
        line_color=get_active_theme()["line"],
        annotation_text="Dataset median position",
        annotation_position="top",
    )
    fig.update_layout(
        title="Selected Numerical Profile — Dataset Percentile Position",
        xaxis_title="Percentile within cleaned dataset",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return base_plot_layout(fig, 500)


# =============================================================================
# DATA-DRIVEN STORYTELLING
# =============================================================================

def key_analytical_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    bmi_means = df.groupby(TARGET, observed=True)["BMI"].mean().reindex(OBESITY_ORDER)
    faf_means = df.groupby(TARGET, observed=True)["FAF"].mean().reindex(OBESITY_ORDER)

    severe = df[TARGET].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"])
    family = (
        df.assign(_severe=severe)
        .groupby("family_history_with_overweight", observed=True)["_severe"]
        .mean()
    )

    favc = (
        df.groupby(TARGET, observed=True)["FAVC"]
        .apply(lambda s: (s == "yes").mean())
        .reindex(OBESITY_ORDER)
    )

    class_counts = df[TARGET].value_counts(normalize=True).reindex(OBESITY_ORDER)

    return [
        {
            "Finding": "The seven target classes remain relatively balanced after duplicate removal.",
            "Evidence": (
                f"Class shares range from {class_counts.min():.2%} to {class_counts.max():.2%} "
                f"across {len(df):,} cleaned observations."
            ),
            "Interpretation": "No single class dominates the cleaned modelling dataset, so accuracy is less likely to be inflated by a majority-class strategy.",
            "Practical Relevance": "Model comparison can focus on balanced multiclass metrics such as weighted/macro F1 alongside accuracy.",
        },
        {
            "Finding": "Body-size measurements strongly separate the ordered obesity categories.",
            "Evidence": (
                f"Mean BMI rises from {bmi_means.iloc[0]:.2f} in {display_label(OBESITY_ORDER[0])} "
                f"to {bmi_means.iloc[-1]:.2f} in {display_label(OBESITY_ORDER[-1])}."
            ),
            "Interpretation": "Height and weight jointly form clear bands across the target classes in this dataset.",
            "Practical Relevance": "These measurements provide strong predictive signal, but their strength should be interpreted cautiously because the target definition is closely related to body size.",
        },
        {
            "Finding": "Family history is strongly associated with the distribution of higher obesity categories.",
            "Evidence": (
                f"Severe obesity classes account for {family.get('yes', np.nan):.1%} of records with family history "
                f"versus {family.get('no', np.nan):.1%} without it."
            ),
            "Interpretation": "The category composition differs substantially between the two family-history groups.",
            "Practical Relevance": "Family history may be useful as a screening/context feature when combined with physical and lifestyle predictors.",
        },
        {
            "Finding": "Reported physical-activity frequency tends to be lower in the highest obesity category.",
            "Evidence": (
                f"Mean FAF is {faf_means.iloc[0]:.2f} in {display_label(OBESITY_ORDER[0])} and "
                f"{faf_means.iloc[-1]:.2f} in {display_label(OBESITY_ORDER[-1])}."
            ),
            "Interpretation": "The distributions overlap, so activity alone does not perfectly separate classes, but the downward tendency adds multivariable signal.",
            "Practical Relevance": "Lifestyle variables can complement body measurements rather than acting as standalone decision rules.",
        },
        {
            "Finding": "Frequent high-calorie food consumption is highly prevalent in several severe obesity classes.",
            "Evidence": (
                f"FAVC = 'yes' is {favc.loc['Obesity_Type_III']:.1%} in Obesity Type III and "
                f"{favc.loc['Normal_Weight']:.1%} in Normal Weight."
            ),
            "Interpretation": "The observed dietary composition differs by obesity category, although the relationship is not perfectly monotonic across all seven classes.",
            "Practical Relevance": "Dietary indicators can support richer risk classification when interpreted together with demographic and physical features.",
        },
    ]


# =============================================================================
# PAGE RENDERERS
# =============================================================================

def render_overview_page(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    render_hero(
        "Obesity Risk Analytics & Classification Dashboard",
        "Interactive exploration and machine-learning classification of obesity levels using demographic, dietary, physical and lifestyle characteristics.",
    )
    render_source_pills(["CRISP-DM", "2,087 cleaned records", "7 classes", "4 ML models", "Held-out evaluation"])

    st.markdown("### Project at a glance")
    render_kpi_grid(
        [
            ("Clean Records", f"{len(clean):,}", "After removing exact duplicates"),
            ("Predictors", "16", "8 numerical + 8 categorical"),
            ("Obesity Classes", f"{clean[TARGET].nunique()}", "Multiclass classification target"),
            ("ML Models", "4", "Logistic, KNN, Random Forest, XGBoost"),
        ]
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("### Project objective")
        st.write(
            "The project classifies individuals into seven obesity-level categories using 16 predictors covering "
            "demographics, physical measurements, eating habits and lifestyle behaviour. The dashboard is designed "
            "to communicate the full analytical chain—from data quality and EDA to model evaluation and a live "
            "interactive prediction in one presentation-focused analytical workflow."
        )

        st.markdown("### Analysis workflow")
        workflow = [
            ("01", "Data", "Load the UCI obesity dataset and understand its 17 variables."),
            ("02", "Preparation", "Assess quality, remove 24 duplicates, split 70:30 with stratification, encode and scale where required."),
            ("03", "EDA", "Explore distributions, obesity-class relationships and appropriate association measures."),
            ("04", "Modelling", "Compare Logistic Regression, KNN, Random Forest and XGBoost using the held-out project evaluation."),
            ("05", "Evaluation", "Inspect class-level performance, ROC behaviour, stability and error patterns."),
            ("06", "Prediction", "Apply the trained classifiers to an interactive 16-feature input form."),
        ]
        for number, title, text in workflow:
            st.markdown(
                f"""
                <div class="flow-step">
                    <div class="flow-num">{number}</div>
                    <div class="flow-content"><b>{title}</b><br><span>{text}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Key analytical findings")
        for insight in key_analytical_insights(clean)[:4]:
            render_insight(insight["Finding"], insight["Evidence"] + " " + insight["Interpretation"])

    st.markdown("### Dataset limitation and responsible interpretation")
    st.warning(
        "The UCI dataset described in the project report combines approximately 23% directly collected observations "
        "with 77% synthetically generated observations. Very strong patterns may therefore partly reflect how the "
        "dataset was constructed. The dashboard treats results as an academic classification analysis; external "
        "validation would be required before any real-world deployment."
    )



def render_data_preparation_page(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    render_hero(
        "Data Preparation",
        "A concise data-quality story showing what was checked, why it mattered, and how the 2,111 raw records became the model-ready dataset used by the project.",
    )

    missing = int(raw.isna().sum().sum())
    duplicates = int(raw.duplicated().sum())
    render_kpi_grid(
        [
            ("Original Records", f"{len(raw):,}", "Original dataset"),
            ("Cleaned Records", f"{len(clean):,}", "After exact duplicate removal"),
            ("Duplicates Removed", f"{duplicates}", "Removed before train/test split"),
            ("Missing Values", f"{missing}", "No imputation required"),
            ("Predictors", "16", "Target excluded before modelling"),
            ("Numerical Features", "8", "Scaled only where model requires it"),
            ("Categorical Features", "8", "One-Hot Encoded"),
            ("Target Classes", f"{clean[TARGET].nunique()}", "Preserved by stratification"),
        ]
    )

    st.markdown("### Preprocessing pipeline")
    steps = [
        ("Raw Dataset", f"Started with {len(raw):,} rows × {raw.shape[1]} columns."),
        ("Missing Value Assessment", f"Found {missing} missing values, so no imputation was introduced."),
        ("Duplicate Detection", f"Found and removed {duplicates} exact duplicate rows before splitting."),
        ("Categorical Consistency", "Verified expected category labels, spelling and capitalization across eight categorical predictors."),
        ("Outlier Validation", "Used IQR flags as diagnostics; plausible bounded/extreme values were retained rather than removed automatically."),
        ("Feature / Target Separation", "Separated 16 predictors from NObeyesdad so the target was never included as an input feature."),
        ("70:30 Stratified Split", "Used stratify = y and random_state = 42, producing 1,460 training and 627 test observations."),
        ("Categorical Encoding", "Applied One-Hot Encoding with drop_first = True, giving the 23-feature encoded model schema."),
        ("Numerical Scaling Where Required", "Logistic Regression and KNN apply StandardScaler within their model pipelines; Random Forest and XGBoost use encoded unscaled predictors."),
        ("Model-Ready Dataset", "The trained classifiers use the project's established 23-feature input schema, which is preserved throughout evaluation and prediction."),
    ]
    for i, (title, text) in enumerate(steps, start=1):
        st.markdown(
            f"""
            <div class="flow-step">
                <div class="flow-num">{i}</div>
                <div class="flow-content"><b>{title}</b><br><span>{text}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Before vs after data-quality summary")
    quality_df = pd.DataFrame(
        [
            ["Records", len(raw), "Remove exact duplicates", len(clean)],
            ["Missing values", missing, "No imputation required", int(clean.isna().sum().sum())],
            ["Duplicate records", duplicates, "Removed before splitting", 0],
            ["Target classes", raw[TARGET].nunique(), "Preserved with stratification", clean[TARGET].nunique()],
            ["Predictor features", 16, "One-Hot Encode categoricals", "23 encoded features"],
        ],
        columns=["Data Quality Item", "Before", "Action", "After"],
    )
    st.dataframe(quality_df, width="stretch", hide_index=True)

    st.markdown("### Train/test split and stratification")
    from sklearn.model_selection import train_test_split

    X = clean.drop(columns=[TARGET, "BMI", "AgeGroup"], errors="ignore")
    y = clean[[TARGET]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    split_cols = st.columns([0.35, 0.65])
    with split_cols[0]:
        render_kpi_grid(
            [
                ("Training Set", f"{len(X_train):,}", "≈70% of cleaned data"),
                ("Testing Set", f"{len(X_test):,}", "≈30% held-out data"),
            ],
            columns_per_row=2,
        )
        render_insight(
            "Why stratification matters",
            "The seven obesity classes remain represented in similar proportions in the full, training and test datasets. This makes model evaluation less sensitive to accidental class-proportion shifts.",
        )

    with split_cols[1]:
        full_pct = y[TARGET].value_counts(normalize=True).reindex(OBESITY_ORDER)
        train_pct = y_train[TARGET].value_counts(normalize=True).reindex(OBESITY_ORDER)
        test_pct = y_test[TARGET].value_counts(normalize=True).reindex(OBESITY_ORDER)
        split_df = pd.DataFrame(
            {
                "Obesity Category": [display_label(c) for c in OBESITY_ORDER],
                "Full Dataset": full_pct.values,
                "Training Set": train_pct.values,
                "Testing Set": test_pct.values,
            }
        ).melt(id_vars="Obesity Category", var_name="Partition", value_name="Share")
        fig = px.bar(
            split_df,
            x="Obesity Category",
            y="Share",
            color="Partition",
            barmode="group",
            color_discrete_map={
                "Full Dataset": "#A9BEC5",
                "Training Set": "#3A8688",
                "Testing Set": "#1F5F75",
            },
            title="Target-class Proportions Across Full, Training and Testing Partitions",
        )
        fig.update_yaxes(tickformat=".0%", title="Share of partition")
        fig.update_xaxes(title="")
        render_plotly(base_plot_layout(fig, 460))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Categorical Consistency", "Outlier Assessment", "Encoding & Scaling", "EDA Enrichment"]
    )

    with tab1:
        rows = []
        for feature in CATEGORICAL_COLUMNS:
            observed = sorted(clean[feature].dropna().astype(str).unique().tolist())
            expected = sorted(EXPECTED_CATEGORIES[feature])
            rows.append(
                {
                    "Feature": feature,
                    "Unique Categories": len(observed),
                    "Observed Labels": ", ".join(observed),
                    "Consistency": "Pass" if observed == expected else "Review",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        render_insight(
            "What this shows",
            "The cleaned data uses the expected category labels without introducing automatic spelling fixes or category merging. This keeps the dashboard consistent with the model training schema.",
        )

    with tab2:
        outlier_df = iqr_outlier_summary(clean)
        st.dataframe(
            outlier_df.style.format(
                {
                    "IQR Lower Bound": "{:.2f}",
                    "IQR Upper Bound": "{:.2f}",
                    "Flagged %": "{:.2%}",
                    "Observed Min": "{:.2f}",
                    "Observed Max": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        render_insight(
            "Why flagged values were retained",
            "The IQR rule is a statistical flag, not an automatic data-error rule. Age, Height, Weight and especially NCP contain IQR-flagged observations, but the project's plausibility/range checks found no obvious invalid entries. Removing them solely because they are statistically unusual would discard valid variability.",
        )

    with tab3:
        encoding_df = pd.DataFrame(
            [
                ["Original predictors", 16, "8 numerical + 8 categorical"],
                ["One-Hot Encoding", 23, "drop_first = True; training/test schema aligned"],
                ["Logistic Regression", 23, "Encoded input → StandardScaler → Logistic Regression"],
                ["KNN", 23, "Encoded input → StandardScaler → KNN"],
                ["Random Forest", 23, "Encoded, unscaled predictors"],
                ["XGBoost", 23, "Encoded, unscaled predictors with fixed booster feature names"],
            ],
            columns=["Stage / Model", "Feature Count", "Data Handling"],
        )
        st.dataframe(encoding_df, width="stretch", hide_index=True)
        render_insight(
            "Leakage-control principle",
            "The train/test split occurs before model-specific scaling. Numerical scaling is applied only where required by the trained classifier.",
        )

    with tab4:
        bmi_means = clean.groupby(TARGET, observed=True)["BMI"].mean().reindex(OBESITY_ORDER)
        age_counts = clean["AgeGroup"].value_counts().reindex(
            ["Teenager (≤19)", "Young Adult (20–30)", "Adult (31–45)", "Senior (46+)"]
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**BMI — EDA-only derived variable**")
            bmi_df = pd.DataFrame(
                {
                    "Obesity Category": [display_label(c) for c in OBESITY_ORDER],
                    "Mean BMI": bmi_means.values,
                }
            )
            st.dataframe(bmi_df.style.format({"Mean BMI": "{:.2f}"}), width="stretch", hide_index=True)
        with col_b:
            st.markdown("**AgeGroup — EDA-only derived variable**")
            age_df = age_counts.rename_axis("Age Group").reset_index(name="Records")
            st.dataframe(age_df, width="stretch", hide_index=True)
        st.info(
            "BMI and AgeGroup enrich descriptive analysis only. They are not added to the 16 trained-model predictors, so this dashboard does not change the existing model feature space or results."
        )



def render_eda_page(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    render_hero(
        "Exploratory Data Analysis",
        "Interactive, question-led exploration of distributions, obesity-class relationships, lifestyle factors and statistically appropriate associations.",
    )

    st.caption("EDA uses the cleaned 2,087-record dataset so the analytical views correspond to the observations used for modelling after duplicate removal.")

    st.markdown("### 1. Obesity class distribution")
    render_plotly(plot_target_distribution(clean), key="eda_target")
    counts = clean[TARGET].value_counts(normalize=True).reindex(OBESITY_ORDER)
    render_insight(
        "Why this matters for evaluation",
        f"The smallest class represents {counts.min():.2%} and the largest {counts.max():.2%} of cleaned records. "
        "Because no single category dominates, a majority-class shortcut is unlikely to explain strong performance. "
        "Accuracy should still be read together with weighted F1, macro F1 and class-level recall."
    )

    st.markdown("### 2. Numerical distribution explorer")
    selected_feature = st.selectbox(
        "Select numerical feature",
        NUMERICAL_COLUMNS,
        index=NUMERICAL_COLUMNS.index("Weight"),
        key="eda_numeric_distribution",
    )
    s = clean[selected_feature].dropna()
    render_kpi_grid(
        [
            ("Mean", f"{s.mean():.2f}", NUMERIC_UNITS[selected_feature]),
            ("Median", f"{s.median():.2f}", NUMERIC_UNITS[selected_feature]),
            ("Std. Dev.", f"{s.std():.2f}", "Sample standard deviation"),
            ("Minimum", f"{s.min():.2f}", NUMERIC_UNITS[selected_feature]),
            ("Maximum", f"{s.max():.2f}", NUMERIC_UNITS[selected_feature]),
            ("Skewness", f"{s.skew():.2f}", distribution_shape(float(s.skew())).capitalize()),
        ],
        columns_per_row=3,
    )
    render_plotly(plot_numeric_distribution(clean, selected_feature), key=f"eda_dist_{selected_feature}")
    render_insight("Analytical insight", numeric_distribution_insight(clean, selected_feature))

    st.markdown("### 3. Obesity level vs numerical features")
    selected_relation = st.selectbox(
        "Select feature to analyse against obesity level",
        ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "Height"],
        index=1,
        key="eda_numeric_relationship",
    )
    render_plotly(plot_numeric_by_class(clean, selected_relation), key=f"eda_box_{selected_relation}")
    render_insight("What this shows", class_trend_insight(clean, selected_relation))

    st.markdown("### 4. Weight–height relationship")
    render_plotly(plot_weight_height(clean), key="eda_weight_height")
    bmi_means = clean.groupby(TARGET, observed=True)["BMI"].mean().reindex(OBESITY_ORDER)
    render_insight(
        "Why the joint pattern matters",
        f"Mean BMI rises from {bmi_means.iloc[0]:.2f} in {display_label(OBESITY_ORDER[0])} to "
        f"{bmi_means.iloc[-1]:.2f} in {display_label(OBESITY_ORDER[-1])}, and the scatter forms visibly ordered bands. "
        "This helps explain why body-size variables are highly predictive. Because obesity labels are closely related to body size and much of the dataset is synthetic, the separation should not be treated as evidence of broader causal relationships."
    )

    st.markdown("### 5. Lifestyle & dietary factor explorer")
    cat_feature = st.selectbox(
        "Select categorical factor",
        CATEGORICAL_COLUMNS,
        index=CATEGORICAL_COLUMNS.index("family_history_with_overweight"),
        key="eda_categorical",
    )
    render_plotly(plot_categorical_relationship(clean, cat_feature), key=f"eda_cat_{cat_feature}")
    render_insight("Key pattern detected", categorical_insight(clean, cat_feature))

    st.markdown("### 6. Relationship & correlation analysis")
    corr_method = st.radio(
        "Numerical correlation method",
        ["Pearson", "Spearman"],
        horizontal=True,
        key="corr_method",
    )
    render_plotly(plot_correlation_heatmap(clean, corr_method), key=f"eda_corr_{corr_method}")

    focus = st.selectbox(
        "Focus correlation with",
        NUMERICAL_COLUMNS,
        index=NUMERICAL_COLUMNS.index("Weight"),
        key="corr_focus",
    )
    corr = clean[NUMERICAL_COLUMNS].corr(method=corr_method.lower())[focus].drop(focus)
    positives = corr[corr > 0].sort_values(ascending=False)
    negatives = corr[corr < 0].sort_values()
    weak = corr.reindex(corr.abs().sort_values().index)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Strongest positive relationship**")
        if not positives.empty:
            st.metric(positives.index[0], f"r = {positives.iloc[0]:.3f}")
        else:
            st.write("No positive numerical relationship in this matrix.")
    with c2:
        st.markdown("**Strongest negative relationship**")
        if not negatives.empty:
            st.metric(negatives.index[0], f"r = {negatives.iloc[0]:.3f}")
        else:
            st.write("No negative numerical relationship in this matrix.")
    with c3:
        st.markdown("**Weakest relationship**")
        if not weak.empty:
            st.metric(weak.index[0], f"r = {weak.iloc[0]:.3f}")

    st.info(
        "The heatmap deliberately includes numerical predictors only. Categorical variables and the multiclass target are not assigned arbitrary integer codes and then interpreted with Pearson correlation. Correlation describes association and does not imply causation."
    )

    numeric_assoc, cat_assoc = target_association_tables(clean)
    assoc_left, assoc_right = st.columns(2)
    with assoc_left:
        st.markdown("#### Numerical features vs obesity category")
        st.caption("Eta-squared: descriptive share of numerical-feature variance occurring between obesity groups.")
        st.dataframe(
            numeric_assoc.style.format({"Eta-squared": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )
    with assoc_right:
        st.markdown("#### Categorical features vs obesity category")
        st.caption("Bias-corrected Cramér's V: categorical association measure from 0 to 1.")
        st.dataframe(
            cat_assoc.style.format({"Cramer's V": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### 7. Key analytical insights")
    for insight in key_analytical_insights(clean):
        with st.expander(insight["Finding"], expanded=False):
            st.markdown(f"**Evidence**  \n{insight['Evidence']}")
            st.markdown(f"**Interpretation**  \n{insight['Interpretation']}")
            st.markdown(f"**Potential practical relevance**  \n{insight['Practical Relevance']}")

    st.markdown("### 8. From EDA to modelling")
    comparison, artifacts, _ = build_comparison_dataframe()
    links = []

    # Data-side evidence
    numeric_assoc_map = numeric_assoc.set_index("Numerical Feature")["Eta-squared"].to_dict()
    cat_assoc_map = cat_assoc.set_index("Categorical Feature")["Cramer's V"].to_dict()

    for model_name in ["Random Forest", "XGBoost"]:
        artifact = artifacts.get(model_name)
        if artifact is None:
            continue
        feature_df, _, _ = model_feature_analysis(model_name, artifact)
        if feature_df is None:
            continue
        grouped = aggregate_feature_importance(feature_df)
        top = grouped.head(5)["Original Feature"].tolist()
        links.append((model_name, top))

    if links:
        for model_name, top_features in links:
            st.markdown(f"**{model_name} — top original-feature signals from available model importance:** {', '.join(top_features)}")
        render_insight(
            "EDA observation → model evidence",
            f"Weight has eta-squared {numeric_assoc_map.get('Weight', np.nan):.3f} against obesity category and is also prominent in tree-based model importance. "
            f"Family history has Cramér's V {cat_assoc_map.get('family_history_with_overweight', np.nan):.3f}; it appears as a contributing feature rather than a causal explanation. "
            f"FAF shows a descriptive decline toward Obesity Type III but lower standalone class separation (eta-squared {numeric_assoc_map.get('FAF', np.nan):.3f}), illustrating why multivariable models combine many weaker signals with stronger physical measurements."
        )
    else:
        render_insight(
            "EDA observation → model evidence",
            "The EDA identifies strong body-size separation and meaningful lifestyle/categorical associations. Tree-based model evidence is shown when the corresponding trained model is available."
        )



def render_model_evaluation_page() -> None:
    render_hero(
        "Model Evaluation",
        "Detailed evaluation of one trained model at a time on the fixed held-out test set, with class-level performance, ROC behaviour, stability and error analysis.",
    )

    model_name = st.selectbox(
        "Select model",
        list(MODEL_REGISTRY.keys()),
        key="model_eval_selector",
    )

    bundle, error = load_evaluation_bundle(model_name)
    if bundle is None:
        st.error(f"{model_name} evaluation results are currently unavailable.")
        return

    metrics = bundle["metrics"]
    render_kpi_grid(
        [
            ("Accuracy", fmt_pct(metrics["Accuracy"]), "Held-out test set"),
            ("Weighted Precision", f"{metrics['Precision']:.4f}", "Support-weighted across classes"),
            ("Weighted Recall", f"{metrics['Recall']:.4f}", "Support-weighted across classes"),
            ("Weighted F1", f"{metrics['Weighted F1']:.4f}", "Primary balanced comparison metric"),
            ("Macro F1", f"{metrics['Macro F1']:.4f}", "Equal weight to all seven classes"),
            ("Weighted OvR ROC-AUC", f"{metrics['ROC-AUC']:.4f}", "Probability discrimination"),
        ],
        columns_per_row=3,
    )

    tabs = st.tabs(["Overview", "Confusion Matrix", "Class Performance", "ROC", "Feature Analysis"])

    with tabs[0]:
        info = MODEL_INFO[model_name]
        left, right = st.columns([0.6, 0.4])
        with left:
            st.markdown(f"### {model_name}")
            st.markdown(f"**Role:** {info['role']}")
            st.markdown(f"**Model family:** {info['family']}")
            st.markdown(f"**Configuration:** {info['configuration']}")
            st.markdown(f"**Strength:** {info['strength']}")
            st.markdown(f"**Limitation:** {info['limitation']}")
        with right:
            st.metric("Misclassified test cases", f"{metrics['Errors']} / {metrics['Test Size']}")
            st.metric("Error rate", fmt_pct(metrics["Error Rate"]))
            cv = CV_RESULTS[model_name]
            st.metric(
                "CV Accuracy",
                f"{cv['CV Accuracy']:.2%}",
                f"± {cv['CV Accuracy Variation']:.2%}",
                delta_color="off",
            )
            st.metric(
                "CV Weighted F1",
                f"{cv['CV F1']:.2%}",
                f"± {cv['CV F1 Variation']:.2%}",
                delta_color="off",
            )
            st.caption(
                "Five-fold cross-validation evaluates how consistently the model performs across "
                "different training subsets, helping assess stability and generalisation beyond a single train–test split."
            )

    with tabs[1]:
        mode = st.radio("View mode", ["Count", "Percentage"], horizontal=True, key=f"cm_mode_{model_name}")
        render_plotly(plot_confusion_matrix(bundle, mode), key=f"cm_{model_name}_{mode}")
        details = confusion_details(bundle)
        pair = details["pair"]
        pair_text = (
            f"{display_label(pair[0])} ↔ {display_label(pair[1])} ({details['pair_count']} combined errors)"
            if pair else "Unavailable"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Best-classified category", display_label(details["best_class"]), f"Recall {details['best_recall']:.1%}")
        c2.metric("Hardest category", display_label(details["hardest_class"]), f"Recall {details['hardest_recall']:.1%}")
        c3.metric("Most frequent confusion pair", pair_text)
        adjacency = "neighbouring" if details["pair_adjacent"] else "non-neighbouring"
        render_insight(
            "Misclassification interpretation",
            f"The most frequent two-way confusion pair is {pair_text}. These are {adjacency} categories in the natural obesity progression. Concentration of errors among adjacent classes is analytically plausible because boundary categories overlap more than extreme categories in the feature space."
        )

    with tabs[2]:
        class_df = class_performance_dataframe(bundle)
        render_plotly(plot_class_performance(class_df), key=f"class_perf_{model_name}")
        st.dataframe(
            class_df[["Obesity Category", "Precision", "Recall", "F1", "Support"]].style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        strongest = class_df.loc[class_df["F1"].idxmax()]
        weakest = class_df.loc[class_df["F1"].idxmin()]
        render_insight(
            "Class-level performance",
            f"The strongest class by F1 is {strongest['Obesity Category']} ({strongest['F1']:.3f}); the weakest is {weakest['Obesity Category']} ({weakest['F1']:.3f}). This shows why a single aggregate accuracy value is not sufficient for evaluating a seven-class model."
        )

    with tabs[3]:
        try:
            roc_details = compute_roc_details(bundle)
            roc_tab1, roc_tab2 = st.tabs(["Per-class ROC", "Micro / Macro ROC"])
            with roc_tab1:
                render_plotly(plot_roc_classes(roc_details), key=f"roc_class_{model_name}")
                auc_table = pd.DataFrame(
                    {
                        "Obesity Category": [display_label(c) for c in OBESITY_ORDER],
                        "Class AUC": [roc_details["class_auc"].get(c, np.nan) for c in OBESITY_ORDER],
                    }
                )
                st.dataframe(
                    auc_table.style.format({"Class AUC": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Each coloured curve is a one-vs-rest class ROC curve; its AUC is shown directly in the legend.")
            with roc_tab2:
                render_plotly(plot_roc_averages(roc_details), key=f"roc_avg_{model_name}")
                st.caption("Micro-average pools all class decisions, while macro-average gives each class equal weight. The dashed diagonal is the random baseline.")
        except Exception:
            st.warning("ROC analysis is unavailable for this model.")

    with tabs[4]:
        feature_df, title, note = model_feature_analysis(model_name, bundle)
        st.markdown(f"### {title}")
        st.write(note)
        if feature_df is None:
            st.info("No supported feature-analysis view is available for this model.")
        else:
            render_plotly(plot_feature_analysis(feature_df, title), key=f"feature_{model_name}")
            st.dataframe(
                feature_df.style.format({"Importance": "{:.5f}"}),
                width="stretch",
                hide_index=True,
            )

def render_model_comparison_page() -> None:
    render_hero(
        "Model Comparison & Selection",
        "Evidence-based comparison across four models using held-out performance, cross-validation stability, class errors, interpretability and practical considerations.",
    )

    comparison, artifacts, model_errors = build_comparison_dataframe()
    if comparison.empty:
        st.error("Model comparison results are currently unavailable.")
        return

    # ============================================================
    # INTERACTIVE MODEL RANKING
    # ============================================================
    comparison = comparison.copy()
    if "Error Rate" not in comparison.columns:
        comparison["Error Rate"] = comparison["Errors"] / comparison["Test Size"]

    ranking_options = {
        "Weighted F1": {"column": "Weighted F1", "ascending": False, "direction": "Higher is better"},
        "Macro F1": {"column": "Macro F1", "ascending": False, "direction": "Higher is better"},
        "Accuracy": {"column": "Accuracy", "ascending": False, "direction": "Higher is better"},
        "Precision": {"column": "Precision", "ascending": False, "direction": "Higher is better"},
        "Recall": {"column": "Recall", "ascending": False, "direction": "Higher is better"},
        "ROC-AUC": {"column": "ROC-AUC", "ascending": False, "direction": "Higher is better"},
        "Error Rate": {"column": "Error Rate", "ascending": True, "direction": "Lower is better"},
    }

    ranking_metric = st.selectbox(
        "Rank Models By",
        list(ranking_options.keys()),
        index=0,
        key="ranking_metric",
    )

    rank_cfg = ranking_options[ranking_metric]
    rank_column = rank_cfg["column"]
    ascending = rank_cfg["ascending"]

    # Stable secondary tie-breakers only determine row order. Models that are
    # identical at the displayed four-decimal precision share the same rank.
    tie_columns = list(dict.fromkeys([rank_column, "Weighted F1", "Accuracy", "ROC-AUC"]))
    tie_ascending = [ascending if col == rank_column else False for col in tie_columns]

    ranked = (
        comparison.sort_values(
            by=tie_columns,
            ascending=tie_ascending,
            kind="stable",
        )
        .reset_index(drop=True)
        .copy()
    )

    displayed_metric = ranked[rank_column].round(4)
    score_order = sorted(displayed_metric.unique(), reverse=not ascending)
    display_rank_map = {value: position + 1 for position, value in enumerate(score_order)}
    ranked["Rank"] = displayed_metric.map(display_rank_map).astype(int)

    best_value = float(ranked.iloc[0][rank_column])
    if ascending:
        ranked["Gap from Best"] = ranked[rank_column] - best_value
    else:
        ranked["Gap from Best"] = best_value - ranked[rank_column]

    top_display_value = round(best_value, 4)
    tied_models = ranked.loc[
        ranked[rank_column].round(4) == top_display_value,
        "Model",
    ].tolist()
    leader_text = " / ".join(tied_models)

    st.caption(
        f"{rank_cfg['direction']}. Weighted F1 is the default because it balances precision "
        "and recall while accounting for class support; it is not treated as universally superior."
    )

    leader_value = f"{best_value:.2%}" if ranking_metric == "Error Rate" else f"{best_value:.4f}"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Highest Ranked by {ranking_metric}", leader_text)
    with c2:
        st.metric(ranking_metric, leader_value)
    with c3:
        st.metric("Ranking Direction", rank_cfg["direction"])

    rank_columns = st.columns(max(1, len(ranked)))
    for col, (_, row) in zip(rank_columns, ranked.iterrows()):
        metric_value = (
            f"{row[rank_column]:.2%}"
            if ranking_metric == "Error Rate"
            else f"{row[rank_column]:.4f}"
        )
        gap_value = (
            f"{row['Gap from Best']:.2%}"
            if ranking_metric == "Error Rate"
            else f"{row['Gap from Best']:.4f}"
        )
        with col:
            render_rank_card(
                rank=int(row["Rank"]),
                model_name=str(row["Model"]),
                score=metric_value,
                gap=gap_value,
                is_best=int(row["Rank"]) == 1,
            )

    chart_ranked = ranked.sort_values(
        rank_column,
        ascending=not ascending,
    ).copy()

    if ranking_metric == "Error Rate":
        chart_text = chart_ranked[rank_column].map(lambda value: f"{value:.2%}")
        chart_hover = "<b>%{y}</b><br>Error Rate: %{x:.2%}<extra></extra>"
    else:
        chart_text = chart_ranked[rank_column].map(lambda value: f"{value:.4f}")
        chart_hover = f"<b>%{{y}}</b><br>{ranking_metric}: %{{x:.4f}}<extra></extra>"

    ranking_fig = go.Figure(
        go.Bar(
            x=chart_ranked[rank_column],
            y=chart_ranked["Model"],
            orientation="h",
            marker_color=[MODEL_COLORS.get(model, "#6B7F93") for model in chart_ranked["Model"]],
            text=chart_text,
            textposition="outside",
            hovertemplate=chart_hover,
        )
    )
    ranking_fig.update_layout(
        title=f"Model Ranking by {ranking_metric}",
        xaxis_title=ranking_metric,
        yaxis_title="",
        showlegend=False,
    )
    if ranking_metric == "Error Rate":
        ranking_fig.update_xaxes(tickformat=".0%")
    else:
        ranking_fig.update_xaxes(range=[0, 1])
    ranking_fig.update_yaxes(categoryorder="array", categoryarray=chart_ranked["Model"].tolist())
    render_plotly(base_plot_layout(ranking_fig, 420), key=f"ranking_{ranking_metric}")

    st.markdown("### Model ranking table")
    ranking_display = ranked[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Weighted F1",
            "Macro F1",
            "ROC-AUC",
            "Error Rate",
            "Gap from Best",
            "Errors",
            "CV Accuracy",
            "CV Accuracy Variation",
            "CV F1",
            "CV F1 Variation",
        ]
    ].copy()

    formatters = {
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "Weighted F1": "{:.4f}",
        "Macro F1": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "Error Rate": "{:.2%}",
        "CV Accuracy": "{:.2%}",
        "CV Accuracy Variation": "± {:.2%}",
        "CV F1": "{:.2%}",
        "CV F1 Variation": "± {:.2%}",
        "Gap from Best": "{:.2%}" if ranking_metric == "Error Rate" else "{:.4f}",
    }
    st.dataframe(
        ranking_display.style.format(formatters),
        width="stretch",
        hide_index=True,
    )

    if len(tied_models) > 1:
        ranking_text = (
            f"{', '.join(tied_models)} share first place by {ranking_metric} at the displayed "
            "four-decimal precision. Secondary tie-breakers are used only to keep row ordering stable."
        )
    elif ranking_metric == "Weighted F1":
        ranking_text = (
            f"{leader_text} ranks first by Weighted F1. This metric balances precision and recall "
            "while accounting for class support."
        )
    elif ranking_metric == "Macro F1":
        ranking_text = (
            f"{leader_text} ranks first by Macro F1, which gives each obesity class equal weight."
        )
    elif ranking_metric == "Accuracy":
        ranking_text = (
            f"{leader_text} ranks first by Accuracy, representing the largest overall share of "
            "correctly classified held-out observations."
        )
    elif ranking_metric == "Precision":
        ranking_text = (
            f"{leader_text} ranks first by weighted Precision across the seven classes."
        )
    elif ranking_metric == "Recall":
        ranking_text = (
            f"{leader_text} ranks first by weighted Recall across the seven classes."
        )
    elif ranking_metric == "ROC-AUC":
        ranking_text = (
            f"{leader_text} ranks first by Weighted One-vs-Rest ROC-AUC, indicating the strongest "
            "probability discrimination under this metric."
        )
    else:
        ranking_text = (
            f"{leader_text} ranks first under Error Rate because it produced the lowest proportion "
            "of misclassified held-out observations."
        )

    render_insight("Ranking insight", ranking_text)
    st.caption(
        "The interactive ranking changes only this comparison view. The project's final model "
        "recommendation below remains based on the broader multi-metric evidence used in the project."
    )

    st.markdown("### Performance comparison")
    metric_cols = ["Accuracy", "Precision", "Recall", "Weighted F1"]
    long = comparison.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
    fig = px.bar(
        long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        color_discrete_map={
            "Accuracy": "#9FB3BA",
            "Precision": "#6B7F93",
            "Recall": "#3A8688",
            "Weighted F1": "#1F5F75",
        },
        title="Held-out Test Performance Across Core Classification Metrics",
    )
    fig.update_yaxes(range=[0, 1], title="Score")
    fig.update_xaxes(title="")
    render_plotly(base_plot_layout(fig, 500), key="comparison_core")
    st.info("ROC-AUC is intentionally not placed on this grouped bar axis; it answers a different discrimination question and is retained in the ranking table and model-evaluation ROC view.")

    st.markdown("### Cross-validation stability vs held-out test performance")
    st.caption(
        "Five-fold cross-validation evaluates how consistently each model performs across different "
        "training subsets, helping assess model stability and generalisation beyond a single train–test split."
    )

    # Model name is the authoritative key. Sorting/ranking never determines CV assignment.
    comparison_by_model = comparison.set_index("Model")
    cv_model_order = [name for name in MODEL_REGISTRY if name in comparison_by_model.index]

    cv_table = pd.DataFrame(
        [
            {
                "Model": model_name,
                "CV Accuracy": CV_RESULTS[model_name]["CV Accuracy"],
                "Variation": CV_RESULTS[model_name]["CV Accuracy Variation"],
                "CV Weighted F1": CV_RESULTS[model_name]["CV F1"],
                "F1 Variation": CV_RESULTS[model_name]["CV F1 Variation"],
                "Test Accuracy": float(comparison_by_model.loc[model_name, "Accuracy"]),
            }
            for model_name in cv_model_order
        ]
    )

    cv_table_display = cv_table.copy()
    for column in ["CV Accuracy", "Variation", "CV Weighted F1", "F1 Variation", "Test Accuracy"]:
        prefix = "± " if column in ["Variation", "F1 Variation"] else ""
        cv_table_display[column] = cv_table_display[column].map(lambda value, p=prefix: f"{p}{value:.2%}")

    st.dataframe(
        cv_table_display,
        width="stretch",
        hide_index=True,
    )

    chart_labels = [MODEL_SHORT_LABELS[name] for name in cv_model_order]
    cv_accuracy_values = [CV_RESULTS[name]["CV Accuracy"] for name in cv_model_order]
    cv_variations = [CV_RESULTS[name]["CV Accuracy Variation"] for name in cv_model_order]
    test_accuracy_values = [float(comparison_by_model.loc[name, "Accuracy"]) for name in cv_model_order]

    cv_custom = np.empty((len(cv_model_order), 2), dtype=object)
    cv_custom[:, 0] = cv_model_order
    cv_custom[:, 1] = cv_variations
    test_custom = np.asarray(cv_model_order, dtype=object).reshape(-1, 1)

    cv_fig = go.Figure()
    cv_fig.add_trace(
        go.Bar(
            x=chart_labels,
            y=cv_accuracy_values,
            name="5-fold CV Accuracy",
            marker_color="#3A8688",
            width=0.34,
            text=[f"{value:.2%}" for value in cv_accuracy_values],
            textposition="outside",
            cliponaxis=False,
            error_y=dict(
                type="data",
                array=cv_variations,
                visible=True,
                color="#3A8688",
                thickness=1.4,
                width=4,
            ),
            customdata=cv_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cross-validation accuracy: %{y:.2%}<br>"
                "Variation: ±%{customdata[1]:.2%}<extra></extra>"
            ),
        )
    )
    cv_fig.add_trace(
        go.Bar(
            x=chart_labels,
            y=test_accuracy_values,
            name="Held-out Test Accuracy",
            marker_color="#1F5F75",
            width=0.34,
            text=[f"{value:.2%}" for value in test_accuracy_values],
            textposition="outside",
            cliponaxis=False,
            customdata=test_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Held-out test accuracy: %{y:.2%}<extra></extra>"
            ),
        )
    )
    cv_fig.update_layout(
        barmode="group",
        bargap=0.18,
        bargroupgap=0.04,
        title="Cross-Validation Accuracy vs Held-out Test Accuracy",
        xaxis_title="",
        yaxis_title="Accuracy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    cv_fig.update_yaxes(range=[0, 1], tickformat=".0%")
    render_plotly(base_plot_layout(cv_fig, 520), key="comparison_cv")

    render_insight(
        "Generalisation insight",
        "XGBoost shows the strongest cross-validation performance and close agreement with its held-out result. "
        "Random Forest also performs strongly, although its larger displayed CV variation indicates greater "
        "fold-to-fold variability. Logistic Regression records lower CV performance than its held-out result, "
        "while KNN shows comparatively close CV and held-out test accuracy."
    )

    st.markdown("### Misclassification comparison")
    error_df = comparison[["Model", "Errors", "Test Size"]].copy()
    error_df["Error Rate"] = error_df["Errors"] / error_df["Test Size"]
    err_fig = px.bar(
        error_df.sort_values("Error Rate"),
        x="Error Rate",
        y="Model",
        orientation="h",
        color="Model",
        color_discrete_map=MODEL_COLORS,
        text=error_df.sort_values("Error Rate")["Error Rate"].map(lambda x: f"{x:.2%}"),
        title="Held-out Misclassification Rate — Lower Is Better",
    )
    err_fig.update_xaxes(tickformat=".0%", title="Error rate")
    err_fig.update_yaxes(title="")
    err_fig.update_layout(showlegend=False)
    render_plotly(base_plot_layout(err_fig, 400), key="comparison_errors")

    if artifacts:
        selected_error_model = st.selectbox(
            "Inspect most common error pairs",
            list(artifacts.keys()),
            key="comparison_error_model",
        )
        art = artifacts[selected_error_model]
        error_pairs = pd.DataFrame({"Actual": art["y_true"], "Predicted": art["y_pred"]})
        error_pairs = error_pairs[error_pairs["Actual"] != error_pairs["Predicted"]]
        if not error_pairs.empty:
            top = (
                error_pairs.groupby(["Actual", "Predicted"]).size().reset_index(name="Count")
                .sort_values("Count", ascending=False).head(6)
            )
            top["Actual"] = top["Actual"].map(display_label)
            top["Predicted"] = top["Predicted"].map(display_label)
            st.dataframe(top, width="stretch", hide_index=True)

    st.markdown("### Strengths, limitations and practical trade-offs")
    tradeoff_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Strength": MODEL_INFO[name]["strength"],
                "Limitation": MODEL_INFO[name]["limitation"],
                "Interpretability": MODEL_INFO[name]["interpretability"],
                "Operational Complexity": MODEL_INFO[name]["operational_complexity"],
            }
            for name in MODEL_REGISTRY
        ]
    )
    st.dataframe(tradeoff_df, width="stretch", hide_index=True)

    st.markdown("### Final model recommendation")
    primary = comparison.sort_values("Weighted F1", ascending=False).iloc[0]
    accuracy_leader = comparison.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    macro_leader = comparison.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
    auc_leader = comparison.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    error_leader = comparison.sort_values("Errors", ascending=True).iloc[0]["Model"]
    cv_leader = comparison.sort_values("CV F1", ascending=False).iloc[0]["Model"]
    leaders = [accuracy_leader, macro_leader, auc_leader, error_leader, cv_leader]
    support_count = sum(name == primary["Model"] for name in leaders)

    st.success(
        f"**Recommended final model: {primary['Model']}**\n\n"
        f"Predictive performance: weighted F1 = **{primary['Weighted F1']:.4f}**, macro F1 = **{primary['Macro F1']:.4f}**, "
        f"accuracy = **{primary['Accuracy']:.2%}**, weighted OvR ROC-AUC = **{primary['ROC-AUC']:.4f}**, and "
        f"**{int(primary['Errors'])}** errors on {int(primary['Test Size'])} held-out cases. The same model also leads "
        f"**{support_count}/5** supporting checks covering accuracy, macro F1, ROC-AUC, lowest errors and CV weighted F1."
    )

    info = MODEL_INFO[primary["Model"]]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_insight("Predictive performance", "Selected primarily by weighted F1, then supported by accuracy, macro F1, ROC-AUC, CV performance and error analysis rather than a synthetic overall score.")
    with c2:
        render_insight("Interpretability", f"{primary['Model']} interpretability is rated {info['interpretability']}. Feature analysis helps explain associations, but it is less transparent than Logistic Regression.")
    with c3:
        render_insight("Operational complexity", f"{primary['Model']} operational complexity is {info['operational_complexity']}. The trained classifier remains practical for an interactive academic presentation.")

    if primary["Model"] == "XGBoost":
        st.info(
            "For the current project results, XGBoost is not selected because of one isolated number: it leads weighted F1, test accuracy, macro F1, weighted ROC-AUC, CV weighted F1 and error count. Its main trade-off is lower intrinsic interpretability and greater model and version sensitivity."
        )



def render_prediction_page(clean: Optional[pd.DataFrame] = None) -> None:
    render_hero(
        "Individual Prediction",
        "Enter one profile, choose a trained model, and inspect the predicted obesity category and full class-probability distribution. This is an academic classification demonstration, not a medical diagnosis.",
    )

    available_models = []
    for name in MODEL_REGISTRY:
        try:
            model, _ = load_prediction_model(name)
            # Validate that the model can accept the established prediction schema.
            get_prediction_columns(name, model)
            available_models.append(name)
        except Exception:
            continue

    if not available_models:
        st.error("No trained model is currently available for prediction.")
        return

    model_name = st.selectbox("Prediction model", available_models, key="prediction_model")

    with st.form("prediction_form"):
        demo_tab, physical_tab, eating_tab, lifestyle_tab = st.tabs(
            ["Demographics", "Physical Measures", "Eating Habits", "Lifestyle"]
        )

        with demo_tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
            with c2:
                age = st.number_input("Age (years)", min_value=14.0, max_value=61.0, value=25.0, step=1.0)
            with c3:
                family_history = st.selectbox("Family history of overweight", ["no", "yes"])

        with physical_tab:
            c1, c2 = st.columns(2)
            with c1:
                height = st.number_input("Height (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01, format="%.2f")
            with c2:
                weight = st.number_input("Weight (kg)", min_value=39.0, max_value=173.0, value=70.0, step=0.5, format="%.1f")
            bmi = weight / (height ** 2)
            st.caption(f"BMI for reference only: **{bmi:.2f}**. BMI is not added to the trained model inputs.")

        with eating_tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                favc = st.selectbox("Frequent high-calorie food (FAVC)", ["no", "yes"])
                fcvc = st.slider("Vegetable consumption (FCVC, 1–3)", 1.0, 3.0, 2.0, 0.1)
                ncp = st.slider("Main meals (NCP, 1–4)", 1.0, 4.0, 3.0, 0.1)
            with c2:
                caec = st.selectbox("Eating between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
                ch2o = st.slider("Water consumption (CH2O, 1–3)", 1.0, 3.0, 2.0, 0.1)
                scc = st.selectbox("Calorie-consumption monitoring (SCC)", ["no", "yes"])
            with c3:
                calc = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])

        with lifestyle_tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                smoke = st.selectbox("Smoking status", ["no", "yes"])
            with c2:
                faf = st.slider("Physical activity frequency (FAF, 0–3)", 0.0, 3.0, 1.0, 0.1)
                tue = st.slider("Technology-use frequency (TUE, 0–2)", 0.0, 2.0, 1.0, 0.1)
            with c3:
                mtrans = st.selectbox(
                    "Transportation (MTRANS)",
                    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
                )

        submitted = st.form_submit_button("Run classification", width="stretch")

        if submitted:
            values = {
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

            try:
                result = make_prediction(model_name, values)
                result["BMI"] = bmi
                result["inputs"] = values
                st.session_state["prediction_result"] = result
            except Exception as exc:
                st.error(f"Prediction could not be completed safely: {exc}")

    result = st.session_state.get("prediction_result")
    if result is None:
        st.info("Complete the form and click **Run classification** to display a prediction.")
        return

    st.markdown(
        '<div class="prediction-heading"><span class="status-dot"></span> Prediction result</div>',
        unsafe_allow_html=True,
    )
    render_kpi_grid(
        [
            ("Predicted Obesity Level", display_label(result["prediction"]), "Highest predicted probability"),
            ("Prediction Confidence", fmt_pct(result["confidence"]), "Maximum class probability"),
            ("BMI (Reference)", f"{result['BMI']:.2f}", "Not an input feature added by this app"),
            ("Selected Model", result["model"], "Trained classifier"),
        ]
    )

    proba_df = pd.DataFrame(
        {
            "Class": result["classes"].astype(str),
            "Probability": result["probabilities"],
        }
    )
    proba_df["Obesity Category"] = proba_df["Class"].map(display_label)
    proba_df["Color"] = proba_df["Class"].map(OBESITY_COLORS)
    proba_df = proba_df.sort_values("Probability", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=proba_df["Probability"],
            y=proba_df["Obesity Category"],
            orientation="h",
            marker_color=proba_df["Color"],
            text=[f"{p:.1%}" for p in proba_df["Probability"]],
            textposition="outside",
            hovertemplate="%{y}<br>Probability: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title="Predicted Probability Across All Seven Classes", xaxis_title="Predicted probability", yaxis_title="")
    fig.update_xaxes(range=[0, max(1.0, float(proba_df["Probability"].max()) * 1.12)], tickformat=".0%")
    render_plotly(base_plot_layout(fig, 480), key="prediction_probability")

    # ------------------------------------------------------------------
    # Cross-model agreement: same input, four independent classifiers.
    # ------------------------------------------------------------------
    agreement_df, probability_matrix = cross_model_prediction_results(result["inputs"])
    if not agreement_df.empty:
        st.markdown("### Model Prediction Agreement")

        display_agreement = agreement_df[["Model", "Prediction", "Confidence"]].copy()
        display_agreement["Confidence"] = display_agreement["Confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(display_agreement, width="stretch", hide_index=True)

        counts = Counter(agreement_df["Raw Prediction"].tolist())
        highest_count = max(counts.values())
        highest_labels = [label for label, count in counts.items() if count == highest_count]
        if len(highest_labels) == 1:
            agreement_text = (
                f"{highest_count} of {len(agreement_df)} models predict "
                f"**{display_label(highest_labels[0])}** for the selected profile."
            )
        else:
            tied = ", ".join(display_label(label) for label in highest_labels)
            agreement_text = (
                f"The highest agreement is {highest_count} of {len(agreement_df)} models, "
                f"with a tie between **{tied}**."
            )
        render_insight("Agreement summary", agreement_text)

        if not probability_matrix.empty:
            render_plotly(
                plot_cross_model_probability_heatmap(probability_matrix),
                key="cross_model_probability_heatmap",
            )

        st.caption(
            "Model agreement reflects consistency among the four classifiers for the selected input "
            "and should not be interpreted as medical certainty."
        )

    # ------------------------------------------------------------------
    # Dataset-relative profile view: percentile positions only.
    # ------------------------------------------------------------------
    if clean is not None:
        st.markdown("### Selected Profile vs Dataset Distribution")
        profile_df = profile_percentile_dataframe(clean, result["inputs"])
        render_plotly(
            plot_profile_percentiles(profile_df),
            key="prediction_profile_percentiles",
        )
        st.caption(
            "Percentiles show the selected numerical values relative to the cleaned dataset for visual comparison only. "
            "They are not additional model inputs or risk scores."
        )

    st.warning(
        "This prediction tool demonstrates machine-learning classification for academic purposes and should not be treated as a medical diagnosis or medical recommendation."
    )

    with st.expander("View submitted input summary"):
        summary = pd.DataFrame(
            [{"Feature": k, "Value": humanize_category(v) if isinstance(v, str) else v} for k, v in result["inputs"].items()]
        )
        st.dataframe(summary, width="stretch", hide_index=True)


# =============================================================================
# SIDEBAR / APP ROUTING
# =============================================================================

raw, clean, dataset_path = get_data()

NAVIGATION_ITEMS = [
    "🏠 Overview",
    "⚙️ Data Preparation",
    "🔍 Exploratory Analysis",
    "🧪 Model Evaluation",
    "🏆 Model Comparison",
    "🔮 Prediction",
]

st.sidebar.markdown("## Obesity Analytics")
st.sidebar.caption("BMDS2003 · Obesity Classification")

page = st.sidebar.radio(
    "Navigation",
    NAVIGATION_ITEMS,
    label_visibility="collapsed",
)

page_did_change = main_page_changed(page)

render_sidebar_context(page, NAVIGATION_ITEMS)
render_sidebar_snapshot(clean)
render_sidebar_active_model(page)

st.sidebar.markdown("---")
render_sidebar_dataset_source()

if page == "🏠 Overview":
    if raw is not None and clean is not None:
        render_overview_page(raw, clean)
elif page == "⚙️ Data Preparation":
    if raw is not None and clean is not None:
        render_data_preparation_page(raw, clean)
elif page == "🔍 Exploratory Analysis":
    if raw is not None and clean is not None:
        render_eda_page(raw, clean)
elif page == "🧪 Model Evaluation":
    render_model_evaluation_page()
elif page == "🏆 Model Comparison":
    render_model_comparison_page()
elif page == "🔮 Prediction":
    render_prediction_page(clean)

# Run only once per main-page change, after the new page has rendered.
if page_did_change:
    scroll_main_view_to_top()

st.markdown("---")
st.caption(
    "BMDS2003 Data Science · Obesity Levels dataset · Analytical dashboard. "
    "EDA-derived BMI/AgeGroup are for interpretation only and are not added to the trained model feature space."
)
