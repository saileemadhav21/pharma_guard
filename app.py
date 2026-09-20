import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="PharmaGuard - Counterfeit Risk Radar", page_icon="💊", layout="wide")


FEATURE_LIST = ["Active_Ingredient_Pct", "Storage_Temp_Deviation", "Distributor_Credibility_Score",
                "Packaging_Discrepancy_Flag", "Unit_Price_Discount_Pct"]


@st.cache_resource
def load():
    # The model is re-trained here at start-up (takes ~2 seconds) instead of loading a saved
    # model file, so it always matches the scikit-learn version installed on the server.
    df = pd.read_csv("data/pharma_batches.csv")
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                       class_weight="balanced_subsample", random_state=42))])
    model.fit(df[FEATURE_LIST], df["Is_Counterfeit"])
    bundle = {
        "model": model,
        "features": FEATURE_LIST,
        "reference": df[df["Is_Counterfeit"] == 0][FEATURE_LIST].median().to_dict(),
        "importances": dict(zip(FEATURE_LIST, model.named_steps["clf"].feature_importances_.tolist())),
    }
    return bundle, json.load(open("metrics.json")), pd.read_csv("data/pharma_batches_scored.csv")


bundle, metrics, scored = load()
model, FEATURES, REF = bundle["model"], bundle["features"], bundle["reference"]

LABELS = {
    "Active_Ingredient_Pct": "Active ingredient (%)",
    "Storage_Temp_Deviation": "Hours outside temp range",
    "Distributor_Credibility_Score": "Distributor credibility",
    "Packaging_Discrepancy_Flag": "Packaging mismatch",
    "Unit_Price_Discount_Pct": "Price discount (%)",
}
LOW_MAX, HIGH_MIN = 0.35, 0.65   # tier cut-offs (checked against out-of-fold results)

PRESETS = {
    "Authentic": dict(api=100.0, temp=1.5, cred=0.82, pack=False, disc=6.0),
    "Storage suspect": dict(api=95.0, temp=20.0, cred=0.50, pack=False, disc=20.0),
    "Counterfeit": dict(api=65.0, temp=12.0, cred=0.30, pack=True, disc=40.0),
}
for k, v in PRESETS["Storage suspect"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("batch_id", "B-DEMO-001")


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ---------- sidebar: batch arrival form ----------
sb = st.sidebar
sb.header("📦 Incoming batch")
sb.caption("Quick demo cases")
c1, c2, c3 = sb.columns(3)
c1.button("Authentic", on_click=apply_preset, args=("Authentic",), width="stretch")
c2.button("Storage", on_click=apply_preset, args=("Storage suspect",), width="stretch")
c3.button("Fake", on_click=apply_preset, args=("Counterfeit",), width="stretch")
sb.divider()
batch_id = sb.text_input("Batch ID", key="batch_id")
distributor = sb.text_input("Distributor (name / ID)", value="Demo Distributor")
region = sb.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"], index=3)
api = sb.slider("Active ingredient (%) - standard 95-105", 20.0, 110.0, step=0.5, key="api")
temp = sb.slider("Hours outside temperature range", 0.0, 72.0, step=0.5, key="temp")
cred = sb.slider("Distributor credibility (0-1)", 0.0, 1.0, step=0.01, key="cred")
pack = sb.toggle("Packaging / barcode mismatch", key="pack")
disc = sb.slider("Price discount offered (%)", 0.0, 70.0, step=0.5, key="disc")

vals = {"Active_Ingredient_Pct": api, "Storage_Temp_Deviation": temp,
        "Distributor_Credibility_Score": cred, "Packaging_Discrepancy_Flag": int(pack),
        "Unit_Price_Discount_Pct": disc}
row = pd.DataFrame([vals])[FEATURES]
risk = float(model.predict_proba(row)[0, 1])


def status(r):
    if r < LOW_MAX:
        return "APPROVED / AUTHENTIC", "#2e7d32", "🟢", "Release to normal supply chain. Routine random sampling only."
    if r < HIGH_MIN:
        return "SUBSTANDARD / STORAGE SUSPECT", "#f9a825", "🟡", "Hold batch. Send sample for laboratory assay and verify cold-chain and transit logs."
    return "CONFISCATE / COUNTERFEIT ALERT", "#c62828", "🔴", "Quarantine immediately. Prioritise for emergency physical inspection and lab verification."


def red_flags():
    f = []
    if api < 95 or api > 105:
        f.append(f"Active ingredient {api:.1f}% is outside the 95-105% standard range.")
    if temp > 12:
        f.append(f"{temp:.1f} hours outside the recommended temperature range (assumed limit: 12 h).")
    if cred < 0.5:
        f.append(f"Distributor credibility {cred:.2f} is below 0.50.")
    if pack:
        f.append("Barcode / label does not match the registered packaging.")
    if disc > 25:
        f.append(f"Unusually high price discount ({disc:.0f}%) - typical of grey-market clearing.")
    return f


def drivers():
    out = {}
    for ftr in FEATURES:
        alt = row.copy()
        alt[ftr] = REF[ftr]
        out[LABELS[ftr]] = risk - model.predict_proba(alt)[0, 1]
    return pd.Series(out).sort_values()


label, color, icon, action = status(risk)
flags = red_flags()

st.title("💊 PharmaGuard - Counterfeit Risk Radar")
st.caption("Early-warning prototype for drug inspectors and hospital procurement officers · Design Thinking project · trained on SYNTHETIC data")

tab1, tab2, tab3, tab4 = st.tabs(["Batch inspection", "Supply chain view", "Model & data", "Design Thinking"])

# ================= TAB 1 =================
with tab1:
    left, right = st.columns([1, 1.2])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100, number={"suffix": "%"},
            title={"text": "Counterfeit risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222"},
                   "steps": [{"range": [0, LOW_MAX * 100], "color": "#a5d6a7"},
                             {"range": [LOW_MAX * 100, HIGH_MIN * 100], "color": "#fff59d"},
                             {"range": [HIGH_MIN * 100, 100], "color": "#ef9a9a"}]}))
        gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(
            f"""<div style="background:{color};padding:16px;border-radius:12px;color:white;text-align:center;font-size:22px;font-weight:700">
            {icon} {label}</div>""", unsafe_allow_html=True)
        st.markdown(f"**Recommended action:** {action}")
    with right:
        st.subheader("Why was it flagged?")
        if flags:
            for f in flags:
                st.markdown(f"- ⚠️ {f}")
        else:
            st.markdown("- ✅ No rule-based red flags on this batch.")
        d = drivers()
        fig = go.Figure(go.Bar(x=d.values * 100, y=d.index, orientation="h",
                               marker_color=["#2e7d32" if v < 0 else "#c62828" for v in d.values]))
        fig.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10),
                          title="Model drivers vs. a typical authentic batch",
                          xaxis_title="Change in risk (percentage points)")
        st.plotly_chart(fig, width="stretch")

    order = f"""DRAFT - LAB INSPECTION REQUEST
Generated: {datetime.now():%d %b %Y %H:%M}
------------------------------------------------
Batch ID:      {batch_id}
Distributor:   {distributor}
Region:        {region}

Active ingredient:        {api:.1f} %
Hours outside temp range: {temp:.1f}
Distributor credibility:  {cred:.2f}
Packaging mismatch:       {'YES' if pack else 'NO'}
Price discount:           {disc:.1f} %

Model risk score: {risk*100:.0f}%
Status:           {label}
Red flags:
{chr(10).join('  - ' + f for f in flags) if flags else '  - none'}

Recommended action: {action}
------------------------------------------------
PROTOTYPE OUTPUT - not a legal order. Must be reviewed and
signed by an authorised drug inspector before any action.
"""
    st.download_button("📄 Generate lab inspection request (draft)", order,
                       file_name=f"inspection_request_{batch_id}.txt", disabled=risk < LOW_MAX)
    st.caption("Enabled only for batches above the Low tier. Draft for demonstration - not a legal document.")

# ================= TAB 2 =================
with tab2:
    st.caption("Scores below are out-of-fold predictions, i.e. each batch was scored by a model that never saw it in training.")
    scored["Tier"] = pd.cut(scored["Risk_Score"], [-0.01, LOW_MAX, HIGH_MIN, 1.01], labels=["Low", "Moderate", "High"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Batches monitored", f"{len(scored):,}")
    k2.metric("High-risk flagged", f"{(scored['Tier'] == 'High').sum():,}")
    k3.metric("Moderate (lab hold)", f"{(scored['Tier'] == 'Moderate').sum():,}")
    k4.metric("Share needing action", f"{(scored['Tier'] != 'Low').mean() * 100:.1f}%")

    heat = scored.pivot_table(index="Region", columns="Drug_Category", values="Risk_Score", aggfunc="mean") * 100
    hfig = px.imshow(heat.round(1), text_auto=True, color_continuous_scale="YlOrRd", aspect="auto",
                     labels=dict(color="Avg risk %"), title="Average risk score (%) by region and drug category")
    st.plotly_chart(hfig, width="stretch")

    dist = (scored.groupby("Distributor_ID")
            .agg(Batches=("Batch_ID", "count"), Avg_Risk_pct=("Risk_Score", lambda s: s.mean() * 100),
                 High_Risk_Batches=("Tier", lambda s: int((s == "High").sum())),
                 Credibility=("Distributor_Credibility_Score", "mean"))
            .sort_values("Avg_Risk_pct", ascending=False).head(10).round(2).reset_index())
    st.subheader("Top 10 riskiest distributors - audit priority list")
    st.dataframe(dist, hide_index=True, width="stretch")

# ================= TAB 3 =================
with tab3:
    st.warning("**Synthetic data notice:** real seizure logs are confidential, so this prototype is trained on synthetic batch logs "
               "generated from documented assumptions (about 8% counterfeit, overlapping distributions, 2% label noise, missing sensor readings). "
               "Scores show the pipeline works - they are NOT real-world accuracy.")
    st.subheader("Data plan")
    st.markdown(
        "- 5,000 synthetic batches, 5 model inputs + Batch ID, region, distributor and drug category.\n"
        "- Only about 8% of batches are counterfeit, so **accuracy is not used** as the headline metric.\n"
        "- Missing sensor readings are imputed with the median inside the model pipeline (fitted on training data only).\n"
        "- Logistic Regression (with scaling) is the baseline; Random Forest is the final model.")
    rows = []
    for name in ["Logistic Regression (baseline)", "Random Forest (final)"]:
        m = metrics[name]
        rows.append({"Model": name, "Recall": m["recall"], "Precision": m["precision"], "F1": m["f1"],
                     "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "PR-AUC (5-fold CV)": m["cv_pr_auc"],
                     "Caught in top 10%": m["recall_top10pct"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Random Forest gives far fewer false alarms at the 0.5 threshold (higher precision); the two models rank batches similarly.")

    cap = pd.Series(metrics["capacity_curve"])
    cfig = go.Figure(go.Scatter(x=[float(i) * 100 for i in cap.index], y=cap.values * 100, mode="lines+markers"))
    cfig.update_layout(title="Inspection capacity: % of counterfeits caught vs % of batches tested (riskiest first)",
                       xaxis_title="% of batches physically tested", yaxis_title="% of counterfeits caught", height=340)
    st.plotly_chart(cfig, width="stretch")

    t = pd.DataFrame(metrics["tiers"]).T.reset_index().rename(
        columns={"index": "Tier", "batches": "Batches", "counterfeit_rate": "Actual counterfeit rate"})
    st.subheader("Do the risk tiers mean something?")
    st.dataframe(t, hide_index=True, width="stretch")

    imp = pd.Series(bundle["importances"]).rename(index=LABELS).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Overall feature importance", labels={"value": "importance", "index": ""}),
                    width="stretch")

# ================= TAB 4 =================
with tab4:
    st.subheader("Design Thinking journey")
    st.markdown(
        "**1 · Empathize** - Drug inspectors and hospital procurement officers cannot test every box; they worry a bad batch will reach patients.\n\n"
        "**2 · Define** - Regulators lack one place to rank incoming batches by risk, so risky shipments slip through while safe ones are audited repeatedly.\n\n"
        "**3 · Ideate** - An explainable risk-scoring model (Random Forest) that shows *why* a batch is flagged, not just a score.\n\n"
        "**4 · Prototype** - This dashboard: batch form, risk gauge, red-flag list, supply-chain heatmap, draft inspection request.\n\n"
        "**5 · Test** - Run the three demo cases, then let 3-5 classmates play inspectors and note where they get confused. Improve and re-test.")
