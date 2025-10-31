"""
Prostate Cancer Early Detection – Streamlit App
-------------------------------------------------
This single-file app lets clinicians:
  • Enter patient info (Age, PSA, DRE result, Family history) and get a calibrated risk estimate
  • Train (or re-train) a model from a CSV using the same pipeline
  • Review evaluation metrics and optimal decision threshold

How to run (in a terminal):
  pip install streamlit scikit-learn pandas numpy joblib
  streamlit run prostate_early_detection_streamlit_app.py

Notes & disclaimer:
  • This tool is for research/decision support only and is NOT a medical diagnosis.
  • Always combine model outputs with clinical judgment and local guidelines.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump, load

warnings.filterwarnings("ignore")

# ======= CONFIG =======
RANDOM_STATE = 42
N_SPLITS_CV = 5
ARTIFACT_MODEL = Path("model.joblib")
ARTIFACT_META = Path("columns.json")
ARTIFACT_METRICS = Path("metrics.txt")

# Expected input names
NUMERIC_COLS = ["Age", "PSA_Level"]
BINARY_COLS = ["DRE_Result", "Family_History"]  # DRE: 0/1; Family: 0/1
TARGET_COLUMN_CATEGORICAL = "Biopsy_Result"      # 'Malignant'/'Benign'
TARGET_COLUMN_NUMERIC = "Cancer"                  # 1/0

DRE_MAP_TXT = {"Normal": 0, "Abnormal": 1, "normal": 0, "abnormal": 1}
FAM_MAP_TXT = {"Yes": 1, "No": 0, "True": 1, "False": 0}

# ========= Data utils ========= #

def load_and_prepare(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "DRE_Result" in df.columns:
        df["DRE_Result"] = (
            df["DRE_Result"].replace(DRE_MAP_TXT).replace({1: 1, 0: 0}).astype(float)
        )
    if "Family_History" in df.columns:
        df["Family_History"] = (
            df["Family_History"].replace(FAM_MAP_TXT).replace({1: 1, 0: 0}).astype(float)
        )

    for col in ["Age", "PSA_Level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    y = None
    if TARGET_COLUMN_NUMERIC in df.columns:
        y = pd.to_numeric(df[TARGET_COLUMN_NUMERIC], errors="coerce")
    elif TARGET_COLUMN_CATEGORICAL in df.columns:
        cat_map = {"Malignant": 1, "Benign": 0, "malignant": 1, "benign": 0}
        y = df[TARGET_COLUMN_CATEGORICAL].map(cat_map)

    keep_cols = [c for c in NUMERIC_COLS + BINARY_COLS if c in df.columns]
    X = df[keep_cols].copy()
    X = X.dropna(how="any")

    if y is not None:
        y = y.loc[X.index]
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask].astype(int)

    return X, y


def build_candidate_models():
    models = {}
    models["logreg"] = LogisticRegression(
        solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE
    )
    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return models


def train_pipeline(df: pd.DataFrame):
    X, y = load_and_prepare(df)
    if y is None:
        raise ValueError(
            "No target column found. Provide either 'Cancer' (1/0) or 'Biopsy_Result' (Malignant/Benign)."
        )

    missing = [c for c in (NUMERIC_COLS + BINARY_COLS) if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )

    candidates = build_candidate_models()

    best_name, best_score = None, -np.inf
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name

    # Fit best and calibrate
    best_est = candidates[best_name]
    pipe = Pipeline([("pre", pre), ("clf", best_est)])
    pipe.fit(X_train, y_train)
    calib = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    calib.fit(X_train, y_train)

    # Evaluate
    probs = calib.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)

    fpr, tpr, thr = roc_curve(y_test, probs)
    youden = tpr - fpr
    best_thr = float(thr[int(np.argmax(youden))])

    def _eval_at(threshold):
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        return tn, fp, fn, tp, classification_report(y_test, preds, digits=3)

    tn_b, fp_b, fn_b, tp_b, rep_b = _eval_at(best_thr)
    tn_5, fp_5, fn_5, tp_5, rep_5 = _eval_at(0.5)

    # Save artifacts
    dump(calib, ARTIFACT_MODEL)
    meta = {
        "feature_order": NUMERIC_COLS + BINARY_COLS,
        "numeric_features": NUMERIC_COLS,
        "binary_features": BINARY_COLS,
        "dre_encoding": {"Normal": 0, "Abnormal": 1},
        "family_history_encoding": {"No": 0, "Yes": 1},
        "target": {"positive_class": 1, "negative_class": 0},
        "selected_model": best_name,
        "cv_auroc_mean": float(best_score),
        "random_state": RANDOM_STATE,
        "best_threshold": best_thr,
    }
    with open(ARTIFACT_META, "w") as f:
        json.dump(meta, f, indent=2)

    with open(ARTIFACT_METRICS, "w") as f:
        f.write(f"Test AUROC: {auroc:.4f}\n")
        f.write(f"Test AUPRC: {auprc:.4f}\n")
        f.write(f"Best threshold: {best_thr:.4f}\n\n")
        f.write("=== Report @ best threshold ===\n")
        f.write(rep_b + "\n")
        f.write("=== Report @ 0.5 threshold ===\n")
        f.write(rep_5 + "\n")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_threshold": best_thr,
        "selected_model": best_name,
        "cv_auroc_mean": best_score,
        "conf_best": {"tn": tn_b, "fp": fp_b, "fn": fn_b, "tp": tp_b},
        "conf_05": {"tn": tn_5, "fp": fp_5, "fn": fn_5, "tp": tp_5},
    }


# ========= UI ========= #

st.set_page_config(page_title="Prostate Cancer Early Detection", page_icon="🧪", layout="centered")

st.title("🧪 Prostate Cancer Early Detection (Decision Support)")

tabs = st.tabs(["Predict", "Train / Update Model", "About"])

# -------- Predict Tab -------- #
with tabs[0]:
    st.subheader("Predict risk for a patient")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=110, value=60, step=1)
        dre_text = st.selectbox("Digital Rectal Exam (DRE)", ["Normal", "Abnormal"], index=0)
    with col2:
        psa = st.number_input("PSA Level (ng/mL)", min_value=0.0, max_value=200.0, value=4.5, step=0.1, format="%.1f")
        fam_text = st.selectbox("Family History of Prostate Cancer", ["No", "Yes"], index=0)

    st.caption("This tool complements, not replaces, clinical judgment.")

    predict_clicked = st.button("Predict Risk", type="primary")

    if predict_clicked:
        if not ARTIFACT_MODEL.exists() or not ARTIFACT_META.exists():
            st.error("Model artifacts not found. Please train a model in the 'Train / Update Model' tab.")
        else:
            model = load(ARTIFACT_MODEL)
            meta = json.load(open(ARTIFACT_META))
            feat_order = meta["feature_order"]
            best_thr = float(meta.get("best_threshold", 0.5))

            x = pd.DataFrame([
                {
                    "Age": float(age),
                    "PSA_Level": float(psa),
                    "DRE_Result": float(DRE_MAP_TXT[dre_text]),
                    "Family_History": float(FAM_MAP_TXT[fam_text]),
                }
            ])
            # Ensure column order
            x = x[feat_order]

            prob = float(model.predict_proba(x)[:, 1][0])
            positive = prob >= best_thr

            st.metric("Estimated probability of cancer", f"{prob:.2%}")
            st.write(f"Decision threshold (Youden's J): **{best_thr:.2f}**")

            if positive:
                st.error("Model classifies as **High Risk** at the current threshold. Consider further diagnostics (e.g., MRI/biopsy) per guidelines.")
            else:
                st.success("Model classifies as **Lower Risk** at the current threshold. Continue routine evaluation and follow-up per guidelines.")

# -------- Train Tab -------- #
with tabs[1]:
    st.subheader("Train or update the model from a CSV dataset")
    st.markdown(
        "Upload a CSV containing columns: **Age, PSA_Level, DRE_Result, Family_History** and either **Cancer (1/0)** or **Biopsy_Result (Malignant/Benign)**."
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write("Sample of uploaded data:")
            st.dataframe(df.head())

            if st.button("Train Model", type="primary"):
                with st.spinner("Training and calibrating..."):
                    metrics = train_pipeline(df)
                st.success("Training complete. Artifacts saved locally (model.joblib, columns.json, metrics.txt).")
                st.json(metrics)
        except Exception as e:
            st.exception(e)

    st.divider()
    if ARTIFACT_METRICS.exists():
        st.download_button(
            label="Download latest metrics.txt",
            data=ARTIFACT_METRICS.read_bytes(),
            file_name="metrics.txt",
            mime="text/plain",
        )

# -------- About Tab -------- #
with tabs[2]:
    st.subheader("About this tool")
    st.markdown(
        """
        **Intended use:** educational/decision-support. It uses logistic regression or random forest
        with cross-validated selection by AUROC and probability calibration. Numeric features are standardized.

        **Inputs**
        - Age (years)
        - PSA level (ng/mL)
        - DRE result: Normal/Abnormal
        - Family history: Yes/No

        **Outputs**
        - Calibrated probability of cancer
        - Threshold chosen via Youden's J (can be revisited locally per practice preference)

        **Data protection:** No data leaves your machine when running locally.
        """
    )
