"""
STEP 1 - DATA PLAN: synthetic pharmaceutical batch logs
-------------------------------------------------------
Real seizure logs / FIRs are confidential, so this project uses a SYNTHETIC
dataset built from documented assumptions (a standard "proof of concept" approach).

Assumptions (write these in your report):
  * ~7-9% of batches are counterfeit/substandard (fraud is rare -> class imbalance)
  * Counterfeits cluster around low-credibility (grey-market) distributors
  * 20% of counterfeits are "high-quality fakes": the API % looks normal, so the
    model must rely on packaging, price and distributor signals instead
  * A small share of AUTHENTIC batches also look odd (bad storage, lab noise)
  * 2% of labels are flipped to mimic lab / paperwork errors
  * Some sensor readings are missing -> imputed later with the median
Because the labels come from rules we wrote, model scores on this data are
OPTIMISTIC. The point is to demonstrate the pipeline, not real-world accuracy.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000
REGIONS = ["North", "South", "East", "West", "Central", "North-East"]
CATEGORIES = ["Antibiotic", "Vaccine", "Analgesic", "Cardiac", "Antidiabetic"]
CAT_RISK = {"Antibiotic": 1.35, "Vaccine": 1.0, "Analgesic": 1.15, "Cardiac": 0.85, "Antidiabetic": 0.9}

# --- 24 distributors: most reputable, 5 grey-market ---
dist = pd.DataFrame({
    "Distributor_ID": [f"D{i:02d}" for i in range(1, 25)],
    "Region": rng.choice(REGIONS, 24),
})
grey = rng.choice(24, 5, replace=False)
dist["base_cred"] = rng.beta(6, 2, 24)
dist.loc[grey, "base_cred"] = rng.beta(2, 5, 5)
dist["volume_w"] = np.where(dist.index.isin(grey), 0.6, 1.0)
dist["volume_w"] /= dist["volume_w"].sum()

idx = rng.choice(24, N, p=dist["volume_w"])
df = pd.DataFrame({
    "Batch_ID": [f"B{100000 + i}" for i in range(N)],
    "Region": dist.loc[idx, "Region"].values,
    "Distributor_ID": dist.loc[idx, "Distributor_ID"].values,
    "Drug_Category": rng.choice(CATEGORIES, N, p=[.25, .15, .25, .2, .15]),
})
c = dist.loc[idx, "base_cred"].values
df["Distributor_Credibility_Score"] = np.clip(c + rng.normal(0, 0.06, N), 0.02, 1.0).round(2)

# --- label (counterfeit / substandard) depends on distributor + drug category ---
p = (0.02 + 0.45 * (1 - c) ** 3) * df["Drug_Category"].map(CAT_RISK).values
y = (rng.random(N) < p).astype(int)

# --- features conditional on the (true) status ---
api = np.where(y == 0, rng.normal(100, 2.2, N), 0.0)
fake = y == 1
hq_fake = fake & (rng.random(N) < 0.20)                 # high-quality fakes
api[fake] = rng.normal(78, 13, fake.sum())
api[hq_fake] = rng.normal(99, 3, hq_fake.sum())
odd_auth = (y == 0) & (rng.random(N) < 0.03)            # authentic but lab-noisy
api[odd_auth] = rng.normal(92, 5, odd_auth.sum())
df["Active_Ingredient_Pct"] = np.clip(api, 20, 108).round(1)

temp = np.where(y == 0, rng.exponential(2.5, N), rng.exponential(7.0, N))
bad_store = (y == 0) & (rng.random(N) < 0.05)           # authentic but poor transit storage
temp[bad_store] = rng.exponential(10, bad_store.sum())
df["Storage_Temp_Deviation"] = np.clip(temp, 0, 72).round(1)

df["Packaging_Discrepancy_Flag"] = np.where(y == 0, rng.random(N) < 0.04, rng.random(N) < 0.50).astype(int)

disc = np.where(y == 0, rng.normal(8, 7, N), rng.normal(30, 13, N))
quiet_fake = fake & (rng.random(N) < 0.30)              # fakes with no visible discount
disc[quiet_fake] = rng.normal(10, 7, quiet_fake.sum())
df["Unit_Price_Discount_Pct"] = np.clip(disc, 0, 70).round(1)

# --- label noise (2%) and missing sensor readings ---
flip = rng.random(N) < 0.02
df["Is_Counterfeit"] = np.where(flip, 1 - y, y)
for col, rate in [("Storage_Temp_Deviation", 0.06), ("Active_Ingredient_Pct", 0.02), ("Unit_Price_Discount_Pct", 0.03)]:
    df.loc[rng.random(N) < rate, col] = np.nan

df.to_csv("data/pharma_batches.csv", index=False)
print(f"Saved data/pharma_batches.csv  shape={df.shape}")
print("\nCounterfeit rate: %.1f%%" % (df["Is_Counterfeit"].mean() * 100))
print("\nMissing values per column:\n", df.isna().sum()[lambda s: s > 0].to_string())
print("\nMean by status:\n", df.groupby("Is_Counterfeit")[[
    "Active_Ingredient_Pct", "Storage_Temp_Deviation", "Distributor_Credibility_Score",
    "Packaging_Discrepancy_Flag", "Unit_Price_Discount_Pct"]].mean().round(2).to_string())
