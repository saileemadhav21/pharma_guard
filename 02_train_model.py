"""
STEP 2 - MODEL TRAINING & VALIDATION
Because only ~8% of batches are counterfeit, plain accuracy is misleading
(predicting "authentic" every time scores ~92%). We report recall, precision,
PR-AUC and "how many counterfeits do we catch if inspectors can test only the
riskiest 10% of batches?".
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/pharma_batches.csv")
FEATURES = ["Active_Ingredient_Pct", "Storage_Temp_Deviation", "Distributor_Credibility_Score",
            "Packaging_Discrepancy_Flag", "Unit_Price_Discount_Pct"]
X, y = df[FEATURES], df["Is_Counterfeit"]


def logreg():   # baseline: needs scaling
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))])


def forest():   # main model (scaling not needed for trees)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                                    class_weight="balanced_subsample", random_state=42))])


def recall_at_top(y_true, score, frac=0.10):
    k = int(len(y_true) * frac)
    top = np.argsort(-score)[:k]
    return float(np.asarray(y_true)[top].sum() / np.asarray(y_true).sum())


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
results = {}
for name, build in [("Logistic Regression (baseline)", logreg), ("Random Forest (final)", forest)]:
    m = build().fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    results[name] = {
        "roc_auc": round(roc_auc_score(y_te, proba), 3),
        "pr_auc": round(average_precision_score(y_te, proba), 3),
        "recall": round(recall_score(y_te, pred), 3),
        "precision": round(precision_score(y_te, pred), 3),
        "f1": round(f1_score(y_te, pred), 3),
        "recall_top10pct": round(recall_at_top(y_te, proba), 3),
        "cv_pr_auc": round(cross_val_score(build(), X, y, cv=skf, scoring="average_precision").mean(), 3),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    print(name, json.dumps(results[name], indent=1))

# ---- out-of-fold risk scores for EVERY batch (honest, not in-sample) ----
oof = cross_val_predict(forest(), X, y, cv=skf, method="predict_proba")[:, 1]
scored = df.copy()
scored["Risk_Score"] = oof.round(4)
scored.to_csv("data/pharma_batches_scored.csv", index=False)

# ---- risk tiers ----
LOW_MAX, HIGH_MIN = 0.35, 0.65
tiers = pd.cut(oof, [-0.01, LOW_MAX, HIGH_MIN, 1.01], labels=["Low", "Moderate", "High"])
tab = pd.DataFrame({"tier": tiers, "y": y}).groupby("tier", observed=True)["y"].agg(["count", "mean"])
results["tiers"] = {t: {"batches": int(r["count"]), "counterfeit_rate": round(float(r["mean"]), 3)} for t, r in tab.iterrows()}
print("\nRisk tiers (out-of-fold):\n", tab.round(3).to_string())

# ---- inspection capacity curve ----
order = np.argsort(-oof)
cum = np.cumsum(y.values[order]) / y.sum()
pts = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
results["capacity_curve"] = {str(p): round(float(cum[max(int(len(y) * p) - 1, 0)]), 3) for p in pts}
print("\nShare of counterfeits caught vs % batches inspected:", results["capacity_curve"])

# ---- final model + reference values for explanations ----
final = forest().fit(X, y)
authentic = df[df["Is_Counterfeit"] == 0][FEATURES]
joblib.dump({"model": final, "features": FEATURES,
             "reference": authentic.median().to_dict(),
             "importances": dict(zip(FEATURES, final.named_steps["clf"].feature_importances_.tolist()))},
            "model.joblib")
json.dump(results, open("metrics.json", "w"), indent=2)
print("\nSaved model.joblib, metrics.json, data/pharma_batches_scored.csv")
