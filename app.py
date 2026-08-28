"""
BMDS2003 Data Science — Obesity Risk Analytics & Classification Dashboard
=========================================================================

This Streamlit application is a presentation/analytics layer for the existing
BMDS2003 project artifacts. It intentionally DOES NOT train or retune any model.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
# Keep this EXACTLY aligned with LabelEncoder.classes_ in the executed notebook.
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
        "model_path": "models/logistic_regression_model.pkl",
        "y_pred_path": "data/lr_y_pred.pkl",
        "y_proba_path": "data/lr_y_pred_proba.pkl",
        "feature_data_candidates": [
            "data/lr_feature_importance.csv",
            "lr_feature_importance.csv",
        ],
    },
    "K-Nearest Neighbours (KNN)": {
        "model_path": "models/knn_model.pkl",
        "y_pred_path": "data/knn_y_pred.pkl",
        "y_proba_path": "data/knn_y_pred_proba.pkl",
        "feature_data_candidates": [
            "data/knn_feature_importance.csv",
            "knn_feature_importance.csv",
        ],
    },
    "Random Forest": {
        "model_path": "models/random_forest_model.pkl",
        "y_pred_path": "data/rf_y_pred.pkl",
        "y_proba_path": "data/rf_y_pred_proba.pkl",
        "feature_data_candidates": [
            "data/rf_feature_importance.csv",
            "rf_feature_importance.csv",
        ],
    },
    "XGBoost": {
        "model_path": "models/xgboost_model.pkl",
        "y_pred_path": "data/xgb_y_pred.pkl",
        "y_proba_path": "data/xgb_y_pred_proba.pkl",
        "feature_data_candidates": [
            "data/xgb_feature_importance.csv",
            "xgb_feature_importance.csv",
        ],
    },
}

X_TEST_PATH = "data/X_test_encoded.pkl"
Y_TEST_PATH = "data/y_test_flat.pkl"
CV_RESULTS_CSV = "data/cv_results.csv"  # optional future export

# Executed DS_4Models notebook reference values. These values are NOT used to
# overwrite held-out test metrics when saved prediction artifacts are present.
# They are used for (1) consistency checks and (2) CV display because the
# current repository plan does not store per-fold CV results as artifacts.
NOTEBOOK_TEST_RESULTS = {
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

# One-standard-deviation values. Random Forest's notebook prints +/- 2*SD,
# so the notebook's displayed 0.0416 / 0.0421 are converted here to ~1 SD.
NOTEBOOK_CV_RESULTS = {
    "Logistic Regression": {
        "CV Accuracy": 0.8568,
        "CV Accuracy Std": 0.0214,
        "CV F1": 0.8542,
        "CV F1 Std": 0.0227,
    },
    "K-Nearest Neighbours (KNN)": {
        "CV Accuracy": 0.8822,
        "CV Accuracy Std": 0.0148,
        "CV F1": 0.8779,
        "CV F1 Std": 0.0165,
    },
    "Random Forest": {
        "CV Accuracy": 0.9253,
        "CV Accuracy Std": 0.0416,
        "CV F1": 0.9264,
        "CV F1 Std": 0.0211,
    },
    "XGBoost": {
        "CV Accuracy": 0.9651,
        "CV Accuracy Std": 0.0105,
        "CV F1": 0.9649,
        "CV F1 Std": 0.0105,
    },
}

NOTEBOOK_TRAIN_ACCURACY = {
    "Logistic Regression": 0.8966,
    "K-Nearest Neighbours (KNN)": 1.0000,
    "Random Forest": 0.9959,
    "XGBoost": 1.0000,
}

MODEL_INFO = {
    "Logistic Regression": {
        "role": "Baseline model",
        "family": "Linear probabilistic classifier",
        "configuration": (
            "max_iter=2000, random_state=42. The deployment Pipeline applies "
            "StandardScaler to the eight numerical predictors internally."
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
            "k=3, Manhattan distance, distance weighting. The deployment Pipeline "
            "applies StandardScaler internally."
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
            "500 trees, max_depth=12, min_samples_split=5, min_samples_leaf=2, "
            "class_weight='balanced'. Uses encoded unscaled predictors."
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
            "learning_rate=0.1, max_depth=5, subsample=1.0. Uses the exact 23-feature "
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
            to { opacity: 1; transform: translateY(0); }
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
            overflow-x: hidden;
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
            animation: appFadeUp .55s ease both;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# =============================================================================
# Dark Mode ONLY — Light Mode remains exactly as defined above.
# =============================================================================

def inject_apple_dark_mode() -> None:
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
           APPLE DARK PALETTE
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
           SIDEBAR — macOS STYLE
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

        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button {
            background: #0A84FF !important;
            color: #FFFFFF !important;
            border: 1px solid #0A84FF !important;
            border-radius: 9px !important;
            box-shadow: none !important;
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

        </style>
        """,
        unsafe_allow_html=True,
    )


inject_apple_dark_mode()


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
        # Apple/macOS-inspired graphite chart theme.
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
            "ObesityDataSet.csv was not found in the repository root or data/ folder."
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
    return base_plot_layout(fig, 500)


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
    return base_plot_layout(fig, 470)


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

    fig = go.Figure()
    for cls in OBESITY_ORDER:
        subset = chart_df[chart_df[TARGET] == cls]
        fig.add_trace(
            go.Box(
                x=subset[feature],
                y=[display_label(cls)] * len(subset),
                name=display_label(cls),
                marker_color=OBESITY_COLORS[cls],
                line_color=OBESITY_COLORS[cls],
                boxpoints="outliers",
                orientation="h",
                showlegend=False,
                hovertemplate=f"{feature}: %{{x:.2f}}<extra>{display_label(cls)}</extra>",
            )
        )

    means = df.groupby(TARGET, observed=True)[feature].mean().reindex(OBESITY_ORDER)
    fig.add_trace(
        go.Scatter(
            x=means.values,
            y=[display_label(c) for c in OBESITY_ORDER],
            mode="markers",
            name="Class mean",
            marker=dict(symbol="diamond", size=9, color=get_active_theme()["marker_contrast"]),
            hovertemplate="Mean: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=numeric_question(feature),
        xaxis_title=f"{feature} — {NUMERIC_UNITS[feature]}",
        yaxis_title="",
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
    return base_plot_layout(fig, 600)


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
    return base_plot_layout(fig, 560)


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
    return base_plot_layout(fig, 600)


# =============================================================================
# MODEL ARTIFACT HELPERS
# =============================================================================

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
            return list(names)

    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if X_test is not None and hasattr(X_test, "columns"):
        return [str(x) for x in X_test.columns]

    raise AttributeError(f"Feature names are unavailable for {model_name}.")


def weighted_ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    y_bin = label_binarize(y_true, classes=classes)
    return float(
        roc_auc_score(
            y_bin,
            y_proba,
            average="weighted",
            multi_class="ovr",
        )
    )


def calculate_saved_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC-AUC": weighted_ovr_auc(y_true, y_proba, classes),
        "Errors": int(np.sum(y_true != y_pred)),
        "Test Size": int(len(y_true)),
    }


def validate_evaluation_artifacts(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> None:
    if y_proba.ndim > 2:
        y_proba = np.squeeze(y_proba)
    if y_proba.ndim != 2:
        raise ValueError("predict_proba artifact must be a 2-D array")
    if len(y_true) != len(y_pred):
        raise ValueError("prediction length does not match y_test")
    if y_proba.shape[0] != len(y_true):
        raise ValueError("probability row count does not match y_test")
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"probability columns ({y_proba.shape[1]}) do not match class count ({len(classes)})"
        )
    if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probability rows do not sum to 1 within tolerance")
    if not set(np.unique(y_true)).issubset(set(classes)):
        raise ValueError("y_test contains labels outside the model probability class order")
    if not set(np.unique(y_pred)).issubset(set(classes)):
        raise ValueError("saved predictions contain labels outside the model class set")


def notebook_match(model_name: str, metrics: Dict[str, float], tolerance: float = 0.0015) -> Tuple[bool, str]:
    expected = NOTEBOOK_TEST_RESULTS.get(model_name)
    if expected is None:
        return True, "No notebook reference configured."

    mismatches = []
    for key in ["Accuracy", "Precision", "Recall", "Weighted F1", "Macro F1", "ROC-AUC"]:
        if abs(metrics[key] - expected[key]) > tolerance:
            mismatches.append(f"{key}: app {metrics[key]:.4f} vs notebook {expected[key]:.4f}")
    if int(metrics["Errors"]) != int(expected["Errors"]):
        mismatches.append(f"Errors: app {metrics['Errors']} vs notebook {expected['Errors']}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "Saved artifacts match the executed notebook reference values within rounding tolerance."


def load_evaluation_artifact(model_name: str) -> Tuple[Optional[Dict], Optional[str]]:
    config = MODEL_REGISTRY[model_name]
    required = [
        config["model_path"],
        X_TEST_PATH,
        Y_TEST_PATH,
        config["y_pred_path"],
        config["y_proba_path"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        return None, "Missing required evaluation file(s): " + ", ".join(missing)

    try:
        model = load_model(config["model_path"])
        X_test = load_joblib_data(X_TEST_PATH)
        y_true = np.asarray(load_joblib_data(Y_TEST_PATH)).ravel().astype(str)
        y_pred = np.asarray(load_joblib_data(config["y_pred_path"])).ravel().astype(str)
        y_proba = np.asarray(load_joblib_data(config["y_proba_path"]), dtype=float)
        if y_proba.ndim > 2:
            y_proba = np.squeeze(y_proba)
        classes = get_model_classes(model_name, model)
        validate_evaluation_artifacts(model_name, y_true, y_pred, y_proba, classes)
        metrics = calculate_saved_metrics(y_true, y_pred, y_proba, classes)
        matches, match_message = notebook_match(model_name, metrics)

        return {
            "model": model,
            "X_test": X_test,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classes": classes,
            "metrics": metrics,
            "notebook_match": matches,
            "notebook_match_message": match_message,
        }, None
    except Exception as exc:
        return None, f"Could not load/validate {model_name} evaluation artifacts: {exc}"


def get_cv_dataframe() -> Tuple[pd.DataFrame, str]:
    if Path(CV_RESULTS_CSV).exists():
        try:
            cv_df = pd.read_csv(CV_RESULTS_CSV)
            required = {"Model", "CV Accuracy", "CV Accuracy Std", "CV F1", "CV F1 Std"}
            if required.issubset(cv_df.columns):
                return cv_df[list(required)].copy(), "Saved CV CSV"
        except Exception:
            pass

    rows = []
    for model_name, values in NOTEBOOK_CV_RESULTS.items():
        rows.append({"Model": model_name, **values})
    return pd.DataFrame(rows), "Executed notebook reference"


def build_comparison_dataframe() -> Tuple[pd.DataFrame, Dict[str, Dict], List[str]]:
    rows = []
    artifacts = {}
    warnings = []

    for model_name in MODEL_REGISTRY:
        artifact, error = load_evaluation_artifact(model_name)
        if artifact is not None:
            artifacts[model_name] = artifact
            row = {"Model": model_name, **artifact["metrics"], "Metric Source": "Saved artifacts"}
            rows.append(row)
            if not artifact["notebook_match"]:
                warnings.append(f"{model_name}: {artifact['notebook_match_message']}")
        else:
            # Presentation fallback: use executed-notebook reference values only, clearly labelled.
            expected = NOTEBOOK_TEST_RESULTS.get(model_name)
            if expected:
                rows.append(
                    {
                        "Model": model_name,
                        **expected,
                        "Test Size": 627,
                        "Metric Source": "Executed notebook reference",
                    }
                )
            warnings.append(f"{model_name}: {error}")

    comparison = pd.DataFrame(rows)
    cv_df, cv_source = get_cv_dataframe()
    if not comparison.empty:
        comparison = comparison.merge(cv_df, on="Model", how="left")
    comparison.attrs["cv_source"] = cv_source
    return comparison, artifacts, warnings


# =============================================================================
# MODEL EVALUATION VISUALISATIONS
# =============================================================================

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
    return base_plot_layout(fig, 650)


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
        fig.add_trace(
            go.Scatter(
                x=roc_details["fpr"][cls],
                y=roc_details["tpr"][cls],
                mode="lines",
                name=f"{display_label(cls)} (AUC {roc_details['class_auc'][cls]:.3f})",
                line=dict(color=OBESITY_COLORS[cls], width=2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline (AUC 0.5)",
            line=dict(color="#7A8B92", dash="dash", width=1.5),
        )
    )
    fig.update_layout(
        title="One-vs-Rest ROC Curves by Obesity Category",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="v", x=1.01, y=1),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return base_plot_layout(fig, 600)


def plot_roc_averages(roc_details: Dict) -> go.Figure:
    micro_fpr, micro_tpr, micro_auc = roc_details["micro"]
    macro_fpr, macro_tpr, macro_auc = roc_details["macro"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=micro_fpr,
            y=micro_tpr,
            mode="lines",
            name=f"Micro-average (AUC {micro_auc:.3f})",
            line=dict(color="#1F5F75", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_fpr,
            y=macro_tpr,
            mode="lines",
            name=f"Macro-average (AUC {macro_auc:.3f})",
            line=dict(color="#3A8688", width=3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color="#7A8B92", dash="dash", width=1.5),
        )
    )
    fig.update_layout(
        title="Micro- and Macro-average ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return base_plot_layout(fig, 520)


def strip_transformer_prefix(name: str) -> str:
    return str(name).split("__", 1)[-1]


def model_feature_analysis(model_name: str, artifact: Dict) -> Tuple[Optional[pd.DataFrame], str, str]:
    model = artifact["model"]
    X_test = artifact["X_test"]

    if model_name == "K-Nearest Neighbours (KNN)":
        path = first_existing_path(MODEL_REGISTRY[model_name]["feature_data_candidates"])
        if path is None:
            return (
                None,
                "KNN has no intrinsic feature importance.",
                "The notebook used permutation importance. Its saved CSV was not found, so no feature ranking is fabricated here.",
            )
        df = pd.read_csv(path)
        possible = [
            "Importance (mean drop in accuracy)",
            "Importance",
            "Mean Importance",
        ]
        importance_col = next((c for c in possible if c in df.columns), None)
        if "Feature" not in df.columns or importance_col is None:
            return None, "Permutation importance unavailable", "The saved KNN CSV format was not recognised."
        out = df[["Feature", importance_col]].copy()
        out.columns = ["Feature", "Importance"]
        out = out.sort_values("Importance", ascending=False).head(15)
        return (
            out,
            "KNN permutation importance (saved notebook analysis)",
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
            note = "The fitted XGBoost model uses tree-based importance_type='gain'. Importance is model association, not causal effect."
        else:
            title = "Random Forest tree-based feature importance"
            note = "Random Forest importance reflects impurity reduction across fitted trees. Importance is model association, not causal effect."
        return out, title, note

    return None, "Feature analysis unavailable", "This model does not expose a supported feature-analysis artifact."


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
    # Prefer the exact shared X_test artifact schema because the trained models
    # were deployed against this 23-column encoded format.
    if Path(X_TEST_PATH).exists():
        try:
            X_test = load_joblib_data(X_TEST_PATH)
            if isinstance(X_test, pd.DataFrame):
                if model_name == "XGBoost" and hasattr(model, "get_booster"):
                    booster_names = model.get_booster().feature_names
                    if booster_names:
                        if list(booster_names) != X_test.columns.tolist():
                            raise ValueError(
                                "XGBoost booster feature names do not match shared X_test_encoded columns."
                            )
                        return list(booster_names)
                return X_test.columns.tolist()
        except Exception:
            pass

    return get_model_feature_names(model_name, model, None)


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
    model_path = MODEL_REGISTRY[model_name]["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Required model file not found: {model_path}")

    model = load_model(model_path)
    encoded_row = build_encoded_prediction_row(model_name, model, input_values)

    raw_pred = model.predict(encoded_row)[0]
    proba = np.asarray(model.predict_proba(encoded_row)[0], dtype=float)

    if model_name == "XGBoost":
        pred_label = str(XGB_CLASS_NAMES[int(raw_pred)])
        proba_classes = XGB_CLASS_NAMES.copy()
    else:
        pred_label = str(raw_pred)
        proba_classes = get_model_classes(model_name, model)

    if len(proba) != len(proba_classes):
        raise ValueError("Prediction probability count does not match model class count.")

    return {
        "model": model_name,
        "prediction": pred_label,
        "confidence": float(proba.max()),
        "classes": proba_classes,
        "probabilities": proba,
        "encoded_columns": list(encoded_row.columns),
    }


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
                f"FAVC='yes' is {favc.loc['Obesity_Type_III']:.1%} in Obesity Type III and "
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
    render_source_pills(["CRISP-DM", "2,087 cleaned records", "7 classes", "4 ML models", "No retraining in app"])

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
            "prediction prototype—without requiring the lecturer to open the notebook."
        )

        st.markdown("### Analysis workflow")
        workflow = [
            ("01", "Data", "Load the UCI obesity dataset and understand its 17 variables."),
            ("02", "Preparation", "Assess quality, remove 24 duplicates, split 70:30 with stratification, encode and scale where required."),
            ("03", "EDA", "Explore distributions, obesity-class relationships and appropriate association measures."),
            ("04", "Modelling", "Compare Logistic Regression, KNN, Random Forest and XGBoost using saved project results."),
            ("05", "Evaluation", "Inspect class-level performance, ROC behaviour, stability and error patterns."),
            ("06", "Prediction", "Apply the saved trained models to an interactive 16-feature input form."),
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
        "dataset was constructed. The dashboard treats results as an academic classification prototype; external "
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
            ("Original Records", f"{len(raw):,}", "Raw UCI dataset"),
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
        ("70:30 Stratified Split", "Used stratify=y and random_state=42, producing 1,460 training and 627 test observations."),
        ("Categorical Encoding", "Applied One-Hot Encoding with drop_first=True, giving the 23-feature encoded model schema."),
        ("Numerical Scaling Where Required", "Logistic Regression and KNN deployment Pipelines scale the eight numerical predictors; Random Forest and XGBoost use encoded unscaled predictors."),
        ("Model-Ready Dataset", "Saved models/predictions use the project's existing 23-feature deployment schema. This dashboard does not retrain or alter it."),
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
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

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
        st.plotly_chart(base_plot_layout(fig, 470), use_container_width=True)

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
            use_container_width=True,
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
                ["One-Hot Encoding", 23, "drop_first=True; training/test schema aligned"],
                ["Logistic Regression", 23, "Encoded input → internal StandardScaler → Logistic Regression"],
                ["KNN", 23, "Encoded input → internal StandardScaler → KNN"],
                ["Random Forest", 23, "Encoded, unscaled predictors"],
                ["XGBoost", 23, "Encoded, unscaled predictors with fixed booster feature names"],
            ],
            columns=["Stage / Model", "Feature Count", "Data Handling"],
        )
        st.dataframe(encoding_df, use_container_width=True, hide_index=True)
        render_insight(
            "Leakage-control principle",
            "The main train/test split occurs before scaling. The deployment models do not refit preprocessing inside Streamlit: they only apply the already-fitted model/pipeline to user input.",
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
            st.dataframe(bmi_df.style.format({"Mean BMI": "{:.2f}"}), use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**AgeGroup — EDA-only derived variable**")
            age_df = age_counts.rename_axis("Age Group").reset_index(name="Records")
            st.dataframe(age_df, use_container_width=True, hide_index=True)
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
    st.plotly_chart(plot_target_distribution(clean), use_container_width=True)
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
    st.plotly_chart(plot_numeric_distribution(clean, selected_feature), use_container_width=True)
    render_insight("Analytical insight", numeric_distribution_insight(clean, selected_feature))

    st.markdown("### 3. Obesity level vs numerical features")
    selected_relation = st.selectbox(
        "Select feature to analyse against obesity level",
        ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "Height"],
        index=1,
        key="eda_numeric_relationship",
    )
    st.plotly_chart(plot_numeric_by_class(clean, selected_relation), use_container_width=True)
    render_insight("What this shows", class_trend_insight(clean, selected_relation))

    st.markdown("### 4. Weight–height relationship")
    st.plotly_chart(plot_weight_height(clean), use_container_width=True)
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
    st.plotly_chart(plot_categorical_relationship(clean, cat_feature), use_container_width=True)
    render_insight("Key pattern detected", categorical_insight(clean, cat_feature))

    st.markdown("### 6. Relationship & correlation analysis")
    corr_method = st.radio(
        "Numerical correlation method",
        ["Pearson", "Spearman"],
        horizontal=True,
        key="corr_method",
    )
    st.plotly_chart(plot_correlation_heatmap(clean, corr_method), use_container_width=True)

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
            use_container_width=True,
            hide_index=True,
        )
    with assoc_right:
        st.markdown("#### Categorical features vs obesity category")
        st.caption("Bias-corrected Cramér's V: categorical association measure from 0 to 1.")
        st.dataframe(
            cat_assoc.style.format({"Cramer's V": "{:.3f}"}),
            use_container_width=True,
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
            "The EDA identifies strong body-size separation and meaningful lifestyle/categorical associations. Tree-based feature evidence will appear here when the saved model artifacts are available to Streamlit."
        )



def render_model_evaluation_page() -> None:
    render_hero(
        "Model Evaluation",
        "Detailed evaluation of one saved model at a time, using the project's held-out test artifacts without retraining or changing probabilities.",
    )

    model_name = st.selectbox(
        "Select model",
        list(MODEL_REGISTRY.keys()),
        key="model_eval_selector",
    )

    artifact, error = load_evaluation_artifact(model_name)
    if artifact is None:
        st.error(error)
        expected = NOTEBOOK_TEST_RESULTS.get(model_name)
        if expected:
            st.info(
                "The executed notebook reference values are available, but class-level/confusion/ROC views require the saved y_test, y_pred and y_pred_proba files."
            )
            st.dataframe(pd.DataFrame([{"Model": model_name, **expected}]), use_container_width=True, hide_index=True)
        return

    metrics = artifact["metrics"]
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

    if artifact["notebook_match"]:
        st.success("Artifact integrity check: saved held-out metrics match the executed notebook reference values within rounding tolerance.")
    else:
        st.warning("Artifact/notebook mismatch detected: " + artifact["notebook_match_message"])

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
            st.metric("Error rate", fmt_pct(metrics["Errors"] / metrics["Test Size"]))
            cv = NOTEBOOK_CV_RESULTS[model_name]
            st.metric("CV accuracy", f"{cv['CV Accuracy']:.4f} ± {cv['CV Accuracy Std']:.4f}")
            st.caption("CV values are taken from the executed notebook unless data/cv_results.csv is provided.")

    with tabs[1]:
        mode = st.radio("View mode", ["Count", "Percentage"], horizontal=True, key=f"cm_mode_{model_name}")
        st.plotly_chart(plot_confusion_matrix(artifact, mode), use_container_width=True)
        details = confusion_details(artifact)
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
        class_df = class_performance_dataframe(artifact)
        st.plotly_chart(plot_class_performance(class_df), use_container_width=True)
        st.dataframe(
            class_df[["Obesity Category", "Precision", "Recall", "F1", "Support"]].style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            use_container_width=True,
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
            roc_details = compute_roc_details(artifact)
            roc_tab1, roc_tab2 = st.tabs(["Per-class ROC", "Micro / Macro ROC"])
            with roc_tab1:
                st.plotly_chart(plot_roc_classes(roc_details), use_container_width=True)
                auc_table = pd.DataFrame(
                    {
                        "Obesity Category": [display_label(c) for c in OBESITY_ORDER],
                        "Class AUC": [roc_details["class_auc"].get(c, np.nan) for c in OBESITY_ORDER],
                    }
                )
                st.dataframe(
                    auc_table.style.format({"Class AUC": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with roc_tab2:
                st.plotly_chart(plot_roc_averages(roc_details), use_container_width=True)
            st.info(
                "ROC probability columns are aligned to each model's actual saved class order. XGBoost uses the fixed LabelEncoder order from the executed notebook; the chart then reorders labels only for presentation. The random baseline is shown at AUC 0.5."
            )
        except Exception as exc:
            st.warning(f"ROC curves could not be generated safely from the saved probabilities: {exc}")

    with tabs[4]:
        feature_df, title, note = model_feature_analysis(model_name, artifact)
        st.markdown(f"### {title}")
        st.write(note)
        if feature_df is None:
            st.info("No supported feature-analysis data is available for this model.")
        else:
            st.plotly_chart(plot_feature_analysis(feature_df, title), use_container_width=True)
            st.dataframe(
                feature_df.style.format({"Importance": "{:.5f}"}),
                use_container_width=True,
                hide_index=True,
            )



def render_model_comparison_page() -> None:
    render_hero(
        "Model Comparison & Selection",
        "Evidence-based comparison across four models using held-out performance, cross-validation stability, class errors, interpretability and deployment considerations.",
    )

    comparison, artifacts, warnings = build_comparison_dataframe()
    if comparison.empty:
        st.error("No saved model metrics or executed-notebook reference values are available.")
        return

    if warnings:
        with st.expander("Artifact availability / consistency notices", expanded=False):
            for warning in warnings:
                st.write("• " + warning)

    ranking_metric = st.selectbox(
        "Rank models by",
        ["Weighted F1", "Macro F1", "Accuracy", "ROC-AUC"],
        index=0,
        key="ranking_metric",
    )
    ranked = comparison.sort_values(ranking_metric, ascending=False).reset_index(drop=True).copy()
    ranked["Rank"] = np.arange(1, len(ranked) + 1)

    st.markdown("### Model ranking table")
    ranking_display = ranked[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Weighted F1",
            "Macro F1",
            "ROC-AUC",
            "Errors",
            "CV Accuracy",
            "CV Accuracy Std",
            "CV F1",
            "CV F1 Std",
            "Metric Source",
        ]
    ].copy()
    st.dataframe(
        ranking_display.style.format(
            {
                "Accuracy": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Macro F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "CV Accuracy": "{:.4f}",
                "CV Accuracy Std": "{:.4f}",
                "CV F1": "{:.4f}",
                "CV F1 Std": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Default ranking uses Weighted F1 because it balances precision and recall while respecting class support. The final recommendation below does not rely on this single metric alone."
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
    st.plotly_chart(base_plot_layout(fig, 520), use_container_width=True)
    st.info("ROC-AUC is intentionally not placed on this grouped bar axis; it answers a different discrimination question and is retained in the ranking table and model-evaluation ROC view.")

    st.markdown("### Cross-validation stability vs held-out test performance")
    cv_fig = go.Figure()
    for _, row in comparison.iterrows():
        model_name = row["Model"]
        cv_fig.add_trace(
            go.Bar(
                x=[model_name],
                y=[row["CV Accuracy"]],
                name="5-fold CV accuracy" if len(cv_fig.data) == 0 else None,
                marker_color="#3A8688",
                error_y=dict(type="data", array=[row["CV Accuracy Std"]], visible=True, color="#3A8688"),
                showlegend=(len(cv_fig.data) == 0),
                hovertemplate=f"{model_name}<br>CV accuracy: {row['CV Accuracy']:.4f}<br>SD: {row['CV Accuracy Std']:.4f}<extra></extra>",
            )
        )
        cv_fig.add_trace(
            go.Bar(
                x=[model_name],
                y=[row["Accuracy"]],
                name="Held-out test accuracy" if len(cv_fig.data) == 1 else None,
                marker_color="#1F5F75",
                showlegend=(len(cv_fig.data) == 1),
                hovertemplate=f"{model_name}<br>Test accuracy: {row['Accuracy']:.4f}<extra></extra>",
            )
        )
    cv_fig.update_layout(
        barmode="group",
        title="Cross-Validation Accuracy (Mean ± 1 SD) vs Held-out Test Accuracy",
        xaxis_title="",
        yaxis_title="Accuracy",
    )
    cv_fig.update_yaxes(range=[0, 1])
    st.plotly_chart(base_plot_layout(cv_fig, 520), use_container_width=True)

    stability = comparison[["Model", "CV Accuracy", "CV Accuracy Std", "Accuracy"]].copy()
    stability["Absolute CV–Test Gap"] = (stability["CV Accuracy"] - stability["Accuracy"]).abs()
    stable_row = stability.sort_values("CV Accuracy Std").iloc[0]
    close_row = stability.sort_values("Absolute CV–Test Gap").iloc[0]
    render_insight(
        "Generalisation evidence",
        f"The smallest CV accuracy SD is {stable_row['CV Accuracy Std']:.4f} for {stable_row['Model']}. "
        f"The closest CV mean to held-out test accuracy is {close_row['Model']} with an absolute gap of {close_row['Absolute CV–Test Gap']:.4f}. "
        "Small fold-to-fold variation and similar CV/test performance provide stronger evidence of stability than test accuracy alone. Cross-validation is not an independent test set."
    )
    st.caption("Random Forest note: its notebook prints ±2×SD. This dashboard displays the one-standard-deviation values (~0.0208 accuracy, ~0.0211 weighted F1) for consistency with the other models.")

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
    st.plotly_chart(base_plot_layout(err_fig, 420), use_container_width=True)

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
            st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("### Strengths, limitations and deployment trade-offs")
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
    st.dataframe(tradeoff_df, use_container_width=True, hide_index=True)

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
        render_insight("Operational complexity", f"{primary['Model']} operational complexity is {info['operational_complexity']}. The current saved artifact is still practical for a Streamlit academic prototype.")

    if primary["Model"] == "XGBoost":
        st.info(
            "For the current executed project results, XGBoost is not selected because of one isolated number: it leads weighted F1, test accuracy, macro F1, weighted ROC-AUC, CV weighted F1 and error count. Its main trade-off is lower intrinsic interpretability and greater deployment/version sensitivity."
        )



def render_prediction_page() -> None:
    render_hero(
        "Individual Prediction Prototype",
        "Enter one profile, choose a saved model, and inspect the predicted obesity category and full class-probability distribution. This is an academic classification demonstration, not a medical diagnosis.",
    )

    available_models = [name for name, cfg in MODEL_REGISTRY.items() if Path(cfg["model_path"]).exists()]
    if not available_models:
        st.error("No required model files were found in models/.")
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

        submitted = st.form_submit_button("Run classification", use_container_width=True)

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
            ("Selected Model", result["model"], "Saved trained artifact"),
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
    st.plotly_chart(base_plot_layout(fig, 500), use_container_width=True)

    st.warning(
        "This prototype demonstrates machine-learning classification for academic purposes and should not be treated as a medical diagnosis or medical recommendation."
    )

    with st.expander("View submitted input summary"):
        summary = pd.DataFrame(
            [{"Feature": k, "Value": humanize_category(v) if isinstance(v, str) else v} for k, v in result["inputs"].items()]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


# =============================================================================
# SIDEBAR / APP ROUTING
# =============================================================================

st.sidebar.markdown("## Obesity Analytics")
st.sidebar.caption("BMDS2003 · CRISP-DM analytical prototype")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "⚙️ Data Preparation",
        "🔍 Exploratory Analysis",
        "🧪 Model Evaluation",
        "🏆 Model Comparison",
        "🔮 Prediction",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
status_model_files = sum(Path(cfg["model_path"]).exists() for cfg in MODEL_REGISTRY.values())
st.sidebar.caption(f"Model files detected: {status_model_files}/4")
st.sidebar.caption(f"Shared test artifact: {'Ready' if Path(X_TEST_PATH).exists() and Path(Y_TEST_PATH).exists() else 'Missing'}")

raw, clean, dataset_path = get_data()

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
    render_prediction_page()

st.markdown("---")
st.caption(
    "BMDS2003 Data Science · Obesity Levels dataset · Analytical prototype. "
    "EDA-derived BMI/AgeGroup are for interpretation only and are not added to the trained model feature space."
)
