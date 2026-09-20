# PharmaGuard - Counterfeit & Substandard Medicine Risk Radar (Design Thinking project)

## Run it
```
pip install -r requirements.txt
python 01_generate_data.py   # Step 1: builds the synthetic batch dataset (data/pharma_batches.csv)
python 02_train_model.py     # Step 2: trains models, saves model.joblib, metrics.json, scored data
streamlit run app.py         # Step 3: dashboard
```
(model.joblib, metrics.json and the data files are already included, so `streamlit run app.py` works immediately.)

## Files
- 01_generate_data.py  - data plan + synthetic data generator (assumptions documented at the top)
- 02_train_model.py    - Logistic Regression baseline vs Random Forest; recall/precision/PR-AUC; out-of-fold scoring
- app.py               - Streamlit dashboard (batch inspection, supply chain view, model & data, design thinking)
- data/                - pharma_batches.csv (raw synthetic), pharma_batches_scored.csv (with out-of-fold risk scores)

## Deploy free
Push this folder to GitHub -> share.streamlit.io -> pick app.py
