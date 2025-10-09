import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GEOparse

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)


# =========================
# 1) Load GEO and build matrices
# =========================

SERIES = "GSE15852"
print(f"Loading {SERIES} ...")
gse = GEOparse.get_GEO(SERIES, destdir=".")

'''
    print out and help understanding the data
'''
gsm = gse.gsms['GSM398143']
attrs = [a for a in dir(gsm) if not a.startswith('_')]
print('attrs:', attrs)
print('metadata:', gsm.metadata)
print('characteristics_ch1: ', gsm.metadata['characteristics_ch1'])
print(gsm.table.shape)
print(gsm.table.head())

# Expression matrix (probes x samples)
expr = gse.pivot_samples("VALUE").apply(pd.to_numeric, errors="coerce")
expr = expr.dropna(axis=0, how="all")

# Optional: detection p-value prefilter (keep probes Present in >= N samples)
try:
    pval = gse.pivot_samples("DETECTION P-VALUE").apply(pd.to_numeric, errors="coerce")
    present = (pval <= 0.04).astype(float)
    min_present_samples = 5
    keep_mask = present.sum(axis=1) >= min_present_samples
    if keep_mask.any():
        before = expr.shape[0]
        expr = expr.loc[keep_mask]
        print(f"After detection prefilter (>= {min_present_samples} Present): {before} -> {expr.shape[0]} features")
except Exception as e:
    print(f"Skip detection prefilter (reason: {e})")

# Remove Affymetrix control probes (AFFX-*) if present
try:
    non_ctrl = ~pd.Index(expr.index.astype(str)).str.startswith("AFFX-")
    if non_ctrl.any():
        before = expr.shape[0]
        expr = expr.loc[non_ctrl]
        print(f"Removed controls (AFFX-*): {before - expr.shape[0]} features")
except Exception:
    pass

# Samples x features
X = expr.T


# =========================
# 2) Robust labels from metadata
# =========================

def _parse_char_lines(lines):
    out = {}
    for s in lines or []:
        if isinstance(s, str) and ":" in s:
            k, v = s.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def label_from_sample(gsm):
    meta = gsm.metadata
    chars = _parse_char_lines(meta.get("characteristics_ch1", []))
    grade = chars.get("grade")
    text = " ".join(
        filter(
            None,
            [
                (grade or ""),
                " ".join(meta.get("source_name_ch1", [])),
                " ".join(meta.get("title", [])),
                " ".join(meta.get("description", [])),
            ],
        )
    ).lower()
    if not text:
        return None
    return "normal" if ("normal" in text and "tumor" not in text) else "tumor"


labels = {sid: label_from_sample(gsm) for sid, gsm in gse.gsms.items()}
keep_samples = [sid for sid in X.index if labels.get(sid) is not None]
X = X.loc[keep_samples]
y = pd.Series(labels).loc[keep_samples]
y_bin = (y == "tumor").astype(int)

print(
    "After preprocessing -> Samples:",
    X.shape[0],
    "Features:",
    X.shape[1],
    "Class counts:",
    y.value_counts().to_dict(),
)


# =========================
# 3) Split: hold out 21 for test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=21, stratify=y_bin, random_state=42
)
print("Train/Test sizes:", X_train.shape, X_test.shape)


# =========================
# 4) Pipelines: feature selection + models
# =========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log2_tf = FunctionTransformer(lambda Z: np.log2(Z + 1.0), validate=False)

pipelines = {
    # Logistic Regression with KBest + PCA
    "LR_KBest_PCA": (
        Pipeline(
            [
                ("log2", log2_tf),
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("kbest", SelectKBest(score_func=f_classif, k=1000)),
                ("scale", StandardScaler(with_mean=True)),
                ("pca", PCA()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", solver="liblinear"
                    ),
                ),
            ]
        ),
        {
            "kbest__k": [300, 1000, 3000],
            "pca__n_components": [0.95, 0.99],
            "clf__C": [0.01, 0.1, 1, 3, 10],
        },
    ),
    # Linear SVC with KBest
    "LinearSVC_KBest": (
        Pipeline(
            [
                ("log2", log2_tf),
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("kbest", SelectKBest(score_func=f_classif, k=1000)),
                ("scale", StandardScaler(with_mean=True)),
                ("svc", LinearSVC(class_weight="balanced", dual="auto", random_state=42)),
            ]
        ),
        {"kbest__k":  ["all", 300, 1000, 3000],  "svc__C": [0.01, 0.1, 1, 3, 10]},
    ),
    # Random Forest with KBest
    "RF_KBest": (
        Pipeline(
            [
                ("log2", log2_tf),
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("kbest", SelectKBest(score_func=f_classif, k=1000)),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        ),
        {
            "kbest__k":  ["all", 300, 1000, 3000], 
            "rf__n_estimators": [300, 600],
            "rf__max_depth": [None, 10, 20],
        },
    ),
}


def estimator_score(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return None


results = []
best_estimators = {}

for name, (pipe, grid_params) in pipelines.items():
    print(f"\nFitting {name} ...")
    grid = GridSearchCV(
        pipe,
        param_grid=grid_params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_estimators[name] = best

    y_pred = best.predict(X_test)
    y_score = estimator_score(best, X_test)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
    pr_auc = (
        average_precision_score(y_test, y_score) if y_score is not None else np.nan
    )
    results.append(
        {
            "model": name,
            "acc": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "cv_auc": float(grid.best_score_),
            "best_params": grid.best_params_,
        }
    )
    print(
        f"{name}: test Acc={acc:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, CV-AUC={grid.best_score_:.3f}"
    )


# =========================
# 5) Summaries and plots
# =========================

res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print("\nModel comparison (sorted by test ROC-AUC):\n", res_df[["model", "acc", "roc_auc", "pr_auc", "cv_auc", "best_params"]])

# Print false positives/negatives with scores/probabilities
print("\nMisclassifications on test (FP/FN) per model:")
for name in res_df["model"]:
    est = best_estimators[name]
    y_pred = est.predict(X_test)
    y_score = estimator_score(est, X_test)
    true = y_test.values
    fp_idx = np.where((true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((true == 1) & (y_pred == 0))[0]
    print(f"{name}: FP={len(fp_idx)}, FN={len(fn_idx)}")
    for i in fp_idx:
        sid = X_test.index[i]
        score_info = f"{y_score[i]:.3f}" if y_score is not None else "NA"
        label = "p_tumor" if hasattr(est, "predict_proba") else "score"
        print(f"  FP: {sid} true=0 predicted=1 {label}={score_info}")
    for i in fn_idx:
        sid = X_test.index[i]
        score_info = f"{y_score[i]:.3f}" if y_score is not None else "NA"
        label = "p_tumor" if hasattr(est, "predict_proba") else "score"
        print(f"  FN: {sid} true=1 predicted=0 {label}={score_info}")

# Bar chart: Accuracy & ROC-AUC
plt.figure()
x = np.arange(len(res_df))
w = 0.35
plt.bar(x - w / 2, res_df["acc"], width=w, label="Accuracy")
plt.bar(x + w / 2, res_df["roc_auc"], width=w, label="ROC-AUC")
plt.xticks(x, res_df["model"], rotation=15)
plt.title("Model Comparison")
plt.legend()
plt.tight_layout()

# ROC & PR curves for best estimators (where scores available)
fig, ax = plt.subplots()
for name in res_df["model"]:
    est = best_estimators[name]
    y_score = estimator_score(est, X_test)
    if y_score is None:
        continue
    RocCurveDisplay.from_predictions(y_test, y_score, name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_title("ROC Curves (Test)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()
for name in res_df["model"]:
    est = best_estimators[name]
    y_score = estimator_score(est, X_test)
    if y_score is None:
        continue
    PrecisionRecallDisplay.from_predictions(y_test, y_score, name=name, ax=ax)
baseline = y_test.mean()
ax.hlines(
    baseline,
    0,
    1,
    colors="gray",
    linestyles="--",
    label=f"Chance (pos rate={baseline:.2f})",
)
ax.set_title("Precision-Recall Curves (Test)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plt.show()
