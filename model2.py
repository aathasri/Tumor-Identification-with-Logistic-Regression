import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
import GEOparse
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# =========================
# 1) Download & load GEO
# =========================
SERIES = "GSE15852"
print(f"Loading {SERIES} ...")
gse = GEOparse.get_GEO(SERIES, destdir=".")
expr = gse.pivot_samples('VALUE')          # features x samples
X = expr.T                                 # samples x features

# =========================
# 2) Label parsing
# =========================
def label_from_sample(gsm):
    grade = gsm.metadata.get("characteristics_ch1", [None])[2].replace("grade: ", "").strip()
    return "normal" if grade == "normal" else "tumor"

labels = {sid: label_from_sample(gsm) for sid, gsm in gse.gsms.items()}
keep = [sid for sid, y in labels.items() if y in {"normal","tumor"} and sid in X.index]
X = X.loc[keep]
y = pd.Series({sid: labels[sid] for sid in keep}).loc[X.index]

print("After label parse -> Samples:", X.shape[0], "Features:", X.shape[1], "Class counts:", y.value_counts().to_dict())

# =========================
# 3) Encode labels
# =========================
y_bin = (y == "tumor").astype(int)

# =========================
# 4) Split dataset (stratified)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.25, stratify=y_bin, random_state=42
)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# =========================
# 5) Preprocess + PCA (fit only on training data)
# =========================
# log2 transform if needed
if X_train.max().max() > 50:
    X_train = np.log2(X_train + 1.0)
    X_test = np.log2(X_test + 1.0)

# drop empty columns
X_train = X_train.dropna(axis=1, how="all")
X_test = X_test[X_train.columns]  # keep same features

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA on training set
pca_full = PCA()
pca_full.fit(X_train_scaled)
pve = pca_full.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

# Plot cumulative PVE
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(pve)+1), cumulative_pve, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative PVE')
plt.title('PCA - Cumulative Proportion of Variance Explained')
plt.grid(True)
plt.tight_layout()
plt.show()

# Number of components to explain 95% variance
n_components_95 = np.argmax(cumulative_pve >= 0.95) + 1
print(f"Number of PCs to keep (95% PVE): {n_components_95}")

# Transform both train and test sets
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert to DataFrame for consistency
X_train = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f"PC{i+1}" for i in range(n_components_95)])
X_test = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f"PC{i+1}" for i in range(n_components_95)])

# =========================
# 6) Define models
# =========================
models = {
    "LogReg": make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    ),
    "LinearSVC(calib)": make_pipeline(
        StandardScaler(with_mean=True),
        CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="isotonic", cv=5)
    ),
    "RandomForest": make_pipeline(
        StandardScaler(with_mean=True),
        RandomForestClassifier(n_estimators=300, random_state=42)
    )
}

# =========================
# 7) Train, CV, Evaluate
# =========================
results = []
probas = {}
preds_map = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    try:
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    except Exception:
        cv_auc = np.nan

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_prob = None
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        probas[name] = y_prob
    except Exception:
        auc = np.nan
        pr_auc = np.nan

    results.append(dict(model=name, acc=acc, roc_auc=auc, pr_auc=pr_auc, cv_auc=cv_auc))
    preds_map[name] = y_pred

# Print summary
print("\nModel comparison:")
for r in results:
    print(f"{r['model']:16s} Acc={r['acc']:.3f}  ROC-AUC={r['roc_auc'] if not np.isnan(r['roc_auc']) else float('nan'):.3f}  PR-AUC={r['pr_auc'] if not np.isnan(r['pr_auc']) else float('nan'):.3f}  CV-AUC(train)={r['cv_auc'] if not np.isnan(r['cv_auc']) else float('nan'):.3f}")

# =========================
# 8) Confusion matrices
# =========================
def plot_confusion(cm, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha='center', va='center')
    plt.tight_layout()

for name, y_pred in preds_map.items():
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, f"Confusion Matrix - {name}")

# =========================
# 9) Bar chart: Accuracy & ROC-AUC
# =========================
labels_m = [r['model'] for r in results]
accs = [r['acc'] for r in results]
aucs = [0 if np.isnan(r['roc_auc']) else r['roc_auc'] for r in results]

plt.figure()
x = np.arange(len(labels_m))
w = 0.35
plt.bar(x - w/2, accs, width=w, label="Accuracy")
plt.bar(x + w/2, aucs, width=w, label="ROC-AUC")
plt.xticks(x, labels_m, rotation=15)
plt.title("Model Comparison")
plt.legend()
plt.tight_layout()

# =========================
# 10) Logistic Regression tuning
# =========================
logreg_grid = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")),
    param_grid={"logisticregression__C": [0.01, 0.1, 1, 3, 10]},
    cv=5, scoring="roc_auc", n_jobs=-1
)
logreg_grid.fit(X_train, y_train)
best_logreg = logreg_grid.best_estimator_
print("\nLogReg best params:", logreg_grid.best_params_, "best CV AUC:", logreg_grid.best_score_)

# =========================
# 11) RandomForest tuning
# =========================
rf_grid = GridSearchCV(
    make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    param_grid={
        "randomforestclassifier__n_estimators": [100, 300, 600],
        "randomforestclassifier__max_depth": [None, 10, 20]
    },
    cv=5, scoring="roc_auc", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("RF best params:", rf_grid.best_params_, "best CV AUC:", rf_grid.best_score_)

# =========================
# 12) ROC & PR curves
# =========================
curve_candidates = {
    "LogReg(best)": best_logreg,
    "RandomForest(best)": best_rf,
}
svc_cal = make_pipeline(
    StandardScaler(with_mean=True),
    CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="isotonic", cv=5)
).fit(X_train, y_train)
curve_candidates["LinearSVC(calib)"] = svc_cal

fig, ax = plt.subplots()
for name, est in curve_candidates.items():
    try:
        prob = est.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax)
    except Exception:
        continue
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Chance")
ax.set_title("ROC Curves (Test)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()
for name, est in curve_candidates.items():
    try:
        prob = est.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, prob, name=name, ax=ax)
    except Exception:
        continue
baseline = y_test.mean()
ax.hlines(baseline, 0, 1, colors='gray', linestyles='--', label=f"Chance (pos rate={baseline:.2f})")
ax.set_title("Precision-Recall Curves (Test)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# =========================
# 13) Interpretability: top LogReg coefficients
# =========================
try:
    final_logreg = best_logreg.named_steps["logisticregression"]
    scaler = best_logreg.named_steps["standardscaler"]
    coef = final_logreg.coef_.ravel()
    genes = X_train.columns
    idx = np.argsort(np.abs(coef))[-15:]
    top_pairs = list(zip(genes[idx], coef[idx]))
    print("\nTop 15 LogReg features (PC, coef):")
    for g, c in top_pairs[::-1]:
        print(f"{g:20s} {c:+.3f}")
except Exception:
    pass

plt.show()
