import pandas as pd
import numpy as np
import re

df = pd.read_csv("lagos_rent_f.csv")
print("Raw shape:", df.shape)

# --- 1. Handle missing Price ---
df = df.dropna(subset=["Price"]).copy()
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

# --- 2. Property type from 'More Info' ---
def get_ptype(s):
    s = s.upper()
    if "MINI FLAT" in s: return "Mini Flat"
    if "SELF CONTAIN" in s: return "Self Contain"
    if "FLAT" in s or "APARTMENT" in s: return "Flat/Apartment"
    if "TERRACED DUPLEX" in s or "TERRACE DUPLEX" in s: return "Terrace Duplex"
    if "SEMI DETACHED" in s: return "Semi-Detached Duplex"
    if "DETACHED DUPLEX" in s or "DUPLEX" in s: return "Detached Duplex"
    if "BLOCKS OF FLATS" in s: return "Block of Flats"
    if "HOUSE" in s: return "House (Other)"
    return "Other"
df["PropertyType"] = df["More Info"].astype(str).apply(get_ptype)

# --- 3. TRUE bedroom count: the 'Bedrooms' column is corrupted (sequential row index,
#     not real bedroom counts - verified: values run 4,5,6,7,8... regardless of listing).
#     Extract real bedroom count via regex from Title/More Info text instead.
def extract_beds(row):
    text = f"{row['Title']} {row['More Info']}".upper()
    m = re.search(r"(\d+)\s*BEDROOM", text)
    if m:
        return int(m.group(1))
    if "SELF CONTAIN" in text or "STUDIO" in text:
        return 1
    return np.nan

df["BedroomsClean"] = df.apply(extract_beds, axis=1)
df = df.drop(columns=["Bedrooms"])  # drop corrupted column

# cap unrealistic bedroom counts (data entry errors)
df.loc[df["BedroomsClean"] > 10, "BedroomsClean"] = np.nan
df["BedroomsClean"] = df["BedroomsClean"].fillna(df["BedroomsClean"].median())

# --- 4. Toilets: impute missing with median by property type ---
df["Toilets"] = df.groupby("PropertyType")["Toilets"].transform(lambda x: x.fillna(x.median()))
df["Toilets"] = df["Toilets"].fillna(df["Toilets"].median())
df.loc[df["Toilets"] > 12, "Toilets"] = df["Toilets"].median()

# --- 5. Neighborhood: group rare categories into 'Other' ---
top_neigh = df["Neighborhood"].value_counts()
keep = top_neigh[top_neigh >= 100].index
df["NeighborhoodGrp"] = np.where(df["Neighborhood"].isin(keep), df["Neighborhood"], "Other")
print("Neighborhood groups kept:", len(keep))

# --- 6. Outlier detection & handling on Price (IQR method), done per property type
#     since price ranges differ hugely between e.g. self-contain vs duplex
def cap_outliers(g):
    q1, q3 = g["Price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = max(q1 - 1.5*iqr, 50_000), q3 + 1.5*iqr
    return g[(g["Price"] >= lo) & (g["Price"] <= hi)]

before = len(df)
bounds = df.groupby("PropertyType")["Price"].quantile([0.25, 0.75]).unstack()
bounds.columns = ["q1", "q3"]
bounds["iqr"] = bounds["q3"] - bounds["q1"]
bounds["lo"] = np.maximum(bounds["q1"] - 1.5 * bounds["iqr"], 50_000)
bounds["hi"] = bounds["q3"] + 1.5 * bounds["iqr"]
df = df.merge(bounds[["lo", "hi"]], left_on="PropertyType", right_index=True)
df = df[(df["Price"] >= df["lo"]) & (df["Price"] <= df["hi"])].drop(columns=["lo", "hi"])
print(f"Outliers removed: {before - len(df)} rows ({(before-len(df))/before:.1%})")

# also drop absurd bedroom/price mismatches (e.g. price < 100k or > 200M annual rent)
df = df[(df["Price"] >= 100_000) & (df["Price"] <= 100_000_000)]

# --- 7. Feature engineering ---
df["PricePerBedroom"] = df["Price"] / df["BedroomsClean"].replace(0, 1)
df["LogPrice"] = np.log1p(df["Price"])
df["BedToiletRatio"] = df["Toilets"] / df["BedroomsClean"].replace(0, 1)
df["IsLuxury"] = ((df["Newly Built"] == 1) & (df["Furnished"] == 1)).astype(int)

print("\nFinal cleaned shape:", df.shape)
print(df[["Price","BedroomsClean","Toilets","Newly Built","Furnished"]].describe())

df.to_csv("lagos_rent_clean.csv", index=False)
print("\nSaved lagos_rent_clean.csv")
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

df = pd.read_csv("lagos_rent_clean.csv")

plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=60, color="#2E86AB", kde=True)
plt.xlabel("Annual Rent (₦)"); plt.title("Distribution of Cleaned Lagos Rental Prices")
plt.tight_layout(); plt.savefig("figs/price_distribution.png", dpi=130); plt.close()

num_cols = ["Price","BedroomsClean","Toilets","Newly Built","Furnished","BedToiletRatio"]
plt.figure(figsize=(7,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Matrix - Numeric Features")
plt.tight_layout(); plt.savefig("figs/correlation_heatmap.png", dpi=130); plt.close()
print("done")
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
df = pd.read_csv("lagos_rent_clean.csv")

features_num = ["BedroomsClean","Toilets","Newly Built","Furnished","BedToiletRatio","IsLuxury"]
features_cat = ["PropertyType","NeighborhoodGrp"]
target = "LogPrice"

X = df[features_num + features_cat]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
], remainder="passthrough")

# ---- Model 1: Linear Regression ----
lr_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
lr_pipe.fit(X_train, y_train)
pred_lr = lr_pipe.predict(X_test)

# ---- Model 2: Random Forest Regressor ----
rf_pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(
    n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
pred_rf = rf_pipe.predict(X_test)

def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log); y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true_log, y_pred_log)  # R2 on log scale (more meaningful, less skew-dominated)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name}: MAE=₦{mae:,.0f} | RMSE=₦{rmse:,.0f} | R²(log)={r2:.3f} | MAPE={mape:.1f}%")
    return dict(name=name, mae=mae, rmse=rmse, r2=r2, mape=mape)

print("=== Regression Model Performance (test set, back-transformed to Naira) ===")
res_lr = evaluate("Linear Regression", y_test, pred_lr)
res_rf = evaluate("Random Forest", y_test, pred_rf)

results_df = pd.DataFrame([res_lr, res_rf])
results_df.to_csv("regression_results.csv", index=False)

# ---- Feature importance (RF) ----
ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(features_cat)
all_names = list(cat_names) + features_num
importances = rf_pipe.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(data=imp_df, y="feature", x="importance", color="#2E86AB")
plt.title("Top 15 Drivers of Lagos Rental Price (Random Forest)")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig("figs/feature_importance.png", dpi=130)
plt.close()

# ---- Actual vs predicted ----
plt.figure(figsize=(6,6))
plt.scatter(np.expm1(y_test), np.expm1(pred_rf), alpha=0.15, s=10, color="#A23B72")
lims = [0, np.expm1(y_test).max()]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("Actual Annual Rent (₦)"); plt.ylabel("Predicted Annual Rent (₦)")
plt.title("Random Forest: Actual vs Predicted Rent")
plt.tight_layout()
plt.savefig("figs/actual_vs_predicted.png", dpi=130)
plt.close()

# ---- Price by property type & neighborhood ----
plt.figure(figsize=(9,5))
order = df.groupby("PropertyType")["Price"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="PropertyType", y="Price", order=order, showfliers=False, palette="viridis")
plt.xticks(rotation=35, ha="right"); plt.ylabel("Annual Rent (₦)")
plt.title("Lagos Rent Distribution by Property Type")
plt.tight_layout()
plt.savefig("figs/price_by_type.png", dpi=130)
plt.close()

plt.figure(figsize=(9,6))
top10 = df.groupby("NeighborhoodGrp")["Price"].median().sort_values(ascending=False).head(10)
sns.barplot(x=top10.values, y=top10.index, color="#F18F01")
plt.xlabel("Median Annual Rent (₦)")
plt.title("Top 10 Most Expensive Lagos Neighborhoods (Median Rent)")
plt.tight_layout()
plt.savefig("figs/top_neighborhoods.png", dpi=130)
plt.close()

print("\nSaved figures to figs/ and regression_results.csv")
"""
TIME SERIES ANALYSIS
---------------------
The scraped Lagos listings dataset is CROSS-SECTIONAL (a snapshot of current
listings, no dates). Nigeria also has no clean, publicly downloadable historical
'Lagos rent index' series (confirmed via search - NBS/CBN only publish general
CPI/inflation, not a city rent index).

To meet the time-series requirement with defensible real-world figures, we
construct an ANNUAL LAGOS RENT INDEX (2016-2026) by deflating the current
average cleaned rent (2026 baseline, from our dataset) backward using Nigeria's
published annual headline inflation rate (source: NBS / Trading Economics /
FocusEconomics), since housing/rent costs in Nigeria move closely with general
CPI inflation (NBS CPI 'Housing & Utilities' sub-index tracks headline closely).
This is clearly disclosed as a constructed/estimated series, not a directly
observed historical Lagos rent index.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

sns.set_style("whitegrid")

df = pd.read_csv("lagos_rent_clean.csv")
current_avg_rent = df["Price"].mean()
print(f"2026 average cleaned annual rent (baseline): ₦{current_avg_rent:,.0f}")

# Nigeria annual headline inflation rate (%), sourced NBS / Trading Economics / FocusEconomics
inflation = {
    2016: 15.7, 2017: 16.5, 2018: 12.1, 2019: 11.4, 2020: 13.25,
    2021: 16.95, 2022: 18.85, 2023: 24.66, 2024: 33.2, 2025: 21.0, 2026: 15.9
}
years = sorted(inflation.keys())

# Build the index by deflating backward from 2026 baseline
rent_index = {2026: current_avg_rent}
for y in sorted(years, reverse=True):
    if y == 2026:
        continue
    rent_index[y] = rent_index[y+1] / (1 + inflation[y+1]/100)

ts = pd.Series({y: rent_index[y] for y in years}).sort_index()
ts.index.name = "Year"
print("\nConstructed Lagos average annual rent index:")
print(ts.round(0))
ts.to_csv("lagos_rent_timeseries.csv", header=["AvgAnnualRent"])

train = ts.iloc[:-2]   # hold out last 2 years for testing
test = ts.iloc[-2:]

# ============ MODEL 3: Holt's Linear (Double) Exponential Smoothing ============
def holt_forecast(series, alpha, beta, n_forecast):
    vals = series.values
    level = [vals[0]]
    trend = [vals[1] - vals[0]]
    fitted = [vals[0]]
    for t in range(1, len(vals)):
        lvl = alpha*vals[t] + (1-alpha)*(level[-1]+trend[-1])
        trd = beta*(lvl-level[-1]) + (1-beta)*trend[-1]
        level.append(lvl); trend.append(trd)
        fitted.append(level[-2]+trend[-2])
    forecast = [level[-1] + (i+1)*trend[-1] for i in range(n_forecast)]
    return np.array(fitted), np.array(forecast), level[-1], trend[-1]

def holt_sse(params, series):
    alpha, beta = params
    if not (0 < alpha < 1 and 0 < beta < 1):
        return 1e18
    fitted, _, _, _ = holt_forecast(series, alpha, beta, 1)
    return np.sum((series.values[1:] - fitted[1:])**2)

opt = minimize(holt_sse, x0=[0.5, 0.3], args=(train,), method="Nelder-Mead")
alpha_opt, beta_opt = opt.x
fitted_holt, forecast_holt_test, lvl, trd = holt_forecast(train, alpha_opt, beta_opt, len(test))
mae_holt = np.mean(np.abs(test.values - forecast_holt_test))
mape_holt = np.mean(np.abs((test.values - forecast_holt_test)/test.values))*100
print(f"\n=== Model 3: Holt's Exponential Smoothing (alpha={alpha_opt:.2f}, beta={beta_opt:.2f}) ===")
print(f"Test MAE=₦{mae_holt:,.0f} | MAPE={mape_holt:.1f}%")

# refit on FULL series for future forecast
fitted_full, future_holt, lvl_f, trd_f = holt_forecast(ts, alpha_opt, beta_opt, 5)
print("5-year forward forecast (Holt):", [f"₦{v:,.0f}" for v in future_holt])

# ============ MODEL 4: AR(2) Autoregressive model (via OLS on lagged values, differenced) ============
d = ts.diff().dropna()  # first difference for stationarity
X_ar = np.column_stack([d.shift(1).values[2:], d.shift(2).values[2:]])
y_ar = d.values[2:]
X_ar_design = np.column_stack([np.ones(len(y_ar)), X_ar])
X_train_ar, y_train_ar = X_ar_design[:-2], y_ar[:-2]
coef, *_ = np.linalg.lstsq(X_train_ar, y_train_ar, rcond=None)
print(f"\n=== Model 4: AR(2) on differenced series === coefficients: {coef.round(3)}")

def ar2_forecast(diffs, coef, n_forecast, last_level):
    d_hist = list(diffs)
    levels = [last_level]
    for _ in range(n_forecast):
        d_next = coef[0] + coef[1]*d_hist[-1] + coef[2]*d_hist[-2]
        d_hist.append(d_next)
        levels.append(levels[-1] + d_next)
    return np.array(levels[1:])

test_forecast_ar = ar2_forecast(d.values[:-2], coef, 2, train.values[-1])
mae_ar = np.mean(np.abs(test.values - test_forecast_ar))
mape_ar = np.mean(np.abs((test.values - test_forecast_ar)/test.values))*100
print(f"Test MAE=₦{mae_ar:,.0f} | MAPE={mape_ar:.1f}%")

future_ar = ar2_forecast(d.values, coef, 5, ts.values[-1])
print("5-year forward forecast (AR(2)):", [f"₦{v:,.0f}" for v in future_ar])

# ============ Comparison table ============
ts_results = pd.DataFrame([
    dict(model="Holt's Exponential Smoothing", mae=mae_holt, mape=mape_holt),
    dict(model="AR(2) Differenced", mae=mae_ar, mape=mape_ar),
])
ts_results.to_csv("timeseries_results.csv", index=False)
print("\n", ts_results)

# ============ Plot ============
future_years = list(range(2027, 2032))
plt.figure(figsize=(10,6))
plt.plot(ts.index, ts.values, marker="o", label="Constructed historical index (2016-2026)", color="#2E86AB")
plt.plot(future_years, future_holt, marker="s", linestyle="--", label="Holt forecast", color="#F18F01")
plt.plot(future_years, future_ar, marker="^", linestyle="--", label="AR(2) forecast", color="#A23B72")
plt.xlabel("Year"); plt.ylabel("Average Annual Rent (₦)")
plt.title("Lagos Average Rent: Historical Index & 5-Year Forecast")
plt.legend(); plt.tight_layout()
plt.savefig("figs/timeseries_forecast.png", dpi=130)
plt.close()

print("\nSaved figs/timeseries_forecast.png, timeseries_results.csv, lagos_rent_timeseries.csv")
