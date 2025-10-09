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
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# Load GEO (GSE7390) and build matrices
# =========================

SERIES = "GSE7390"
print(f"Loading {SERIES} ...")
gse = GEOparse.get_GEO(SERIES, destdir=".")

'''
    print out and help understanding the data
'''
gsm = gse.gsms['GSM177950']
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
    min_present_samples = 10
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
# Labels: ER status (0/1) from characteristics_ch1
# =========================

def parse_chars(lines):
    out = {}
    for s in lines or []:
        if isinstance(s, str) and ":" in s:
            k, v = s.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def label_from_sample_er(gsm):
    chars = parse_chars(gsm.metadata.get("characteristics_ch1", []))
    er = chars.get("er")
    if er is None:
        return None
    er = str(er).strip().lower()
    if er in {"1", "pos", "+", "positive"}:
        return 1
    if er in {"0", "neg", "-", "negative"}:
        return 0
    try:
        return int(float(er))
    except Exception:
        return None


labels = {sid: label_from_sample_er(gsm) for sid, gsm in gse.gsms.items()}
keep_samples = [sid for sid in X.index if labels.get(sid) in (0, 1)]
X = X.loc[keep_samples]
y = pd.Series(labels).loc[keep_samples].astype(int)

print(
    "After preprocessing -> Samples:",
    X.shape[0],
    "Features:",
    X.shape[1],
    "Class counts:",
    y.value_counts().to_dict(),
)


# =========================
# Split: hold out 40 for test (≈20%)
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=40, stratify=y, random_state=42
)
print("Train/Test sizes:", X_train.shape, X_test.shape)


# =========================
# Pipelines: feature selection + models
# =========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class AutoLog2(BaseEstimator, TransformerMixin):
    """Apply log2(x+1) when data appear on linear scale; pass-through if already log-like.
    Heuristic: if 95th percentile > threshold (default 20), treat as linear-scale.
    Also clips negatives to 0 before log to avoid warnings.
    """
    def __init__(self, threshold=20.0, clip=True):
        self.threshold = threshold
        self.clip = clip
        self.apply_ = None

    def fit(self, X, y=None):
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        x = np.asarray(Xv, dtype=float)
        finite = np.isfinite(x)
        vals = x[finite]
        if vals.size == 0:
            self.apply_ = False
        else:
            q95 = np.nanpercentile(vals, 95)
            self.apply_ = bool(q95 > self.threshold)
        return self

    def transform(self, X):
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        x = np.asarray(Xv, dtype=float)
        if self.apply_:
            if self.clip:
                x = np.clip(x, 0, None)
            x = np.log2(x + 1.0)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(x, index=X.index, columns=X.columns)
        return x

log2_tf = AutoLog2()

pipelines = {
    "LR_KBest_PCA": (
        Pipeline(
            [
                ("log2", log2_tf),
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("kbest", SelectKBest(score_func=f_classif, k=1000)),
                ("scale", StandardScaler(with_mean=True)),
                ("pca", PCA()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        {"kbest__k": [500, 1000, 3000], "pca__n_components": [0.95, 0.99], "clf__C": [0.01, 0.1, 1, 3, 10]},
    ),
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
        {"kbest__k": ["all", 1000, 3000], "svc__C": [0.01, 0.1, 1, 3, 10]},
    ),
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
        {"kbest__k": [1000, 3000, "all"], "rf__n_estimators": [300, 600], "rf__max_depth": [None, 10, 20]},
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
    pr_auc = average_precision_score(y_test, y_score) if y_score is not None else np.nan
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
    print(f"{name}: test Acc={acc:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, CV-AUC={grid.best_score_:.3f}")


# =========================
# Summaries and plots
# =========================

res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print("\nModel comparison (sorted by test ROC-AUC):\n", res_df[["model", "acc", "roc_auc", "pr_auc", "cv_auc", "best_params"]])

# Bar chart: Accuracy & ROC-AUC
plt.figure()
x = np.arange(len(res_df))
w = 0.35
plt.bar(x - w / 2, res_df["acc"], width=w, label="Accuracy")
plt.bar(x + w / 2, res_df["roc_auc"], width=w, label="ROC-AUC")
plt.xticks(x, res_df["model"], rotation=15)
plt.title("Model Comparison (ER status)")
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
ax.hlines(baseline, 0, 1, colors="gray", linestyles="--", label=f"Chance (pos rate={baseline:.2f})")
ax.set_title("Precision-Recall Curves (Test)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plt.show()
