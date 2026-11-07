"""Train and evaluate the food-waste prediction and demand models; save figures and metrics."""
import json, time, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
t0 = time.time()
df = pd.read_csv("data/food_waste.csv", parse_dates=["date"])
print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")
print("Missing values:", int(df.isna().sum().sum()), "| Duplicate rows:", int(df.duplicated().sum()))

# ---------- cleaning ----------
df = df.drop_duplicates().dropna()
df = df[(df.prepared_qty >= 0) & (df.sold_qty >= 0) & (df.waste_qty >= 0)]
df = df.sort_values(["item", "date"]).reset_index(drop=True)

# ---------- feature engineering (past information only) ----------
g = df.groupby("item")
df["lag1_sold"] = g["sold_qty"].shift(1)
df["lag1_waste"] = g["waste_qty"].shift(1)
df["roll7_sold"] = g["sold_qty"].transform(lambda s: s.shift(1).rolling(7).mean())
df["roll7_waste"] = g["waste_qty"].transform(lambda s: s.shift(1).rolling(7).mean())
df = df.dropna().reset_index(drop=True)
df["waste_pct"] = df.waste_qty / df.prepared_qty * 100
print(f"After cleaning and lag features: {len(df)} records")
print(f"Overall waste: {df.waste_qty.sum()} of {df.prepared_qty.sum()} prepared servings "
      f"({df.waste_qty.sum()/df.prepared_qty.sum()*100:.1f}%)")

items = sorted(df.item.unique())
X_all = pd.get_dummies(df, columns=["item"], dtype=int)
base_cols = ["day_of_week", "month", "is_weekend", "is_holiday", "is_event", "temperature", "rain",
             "expected_footfall", "unit_cost", "price"]
item_cols = [c for c in X_all.columns if c.startswith("item_")]
waste_feats = base_cols + ["prepared_qty", "lag1_sold", "lag1_waste", "roll7_sold", "roll7_waste"] + item_cols
demand_feats = base_cols + ["lag1_sold", "roll7_sold"] + item_cols

# time-based split: first 80% of dates for training, last 20% for testing
cut = df.date.quantile(0.80)
tr = (df.date <= cut).values; te = ~tr
print(f"Train: {tr.sum()} records (to {cut.date()}) | Test: {te.sum()} records")

def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5, R2=r2_score(y, p))

# ---------- Task 1: waste prediction ----------
y = X_all["waste_qty"]
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3,
                                           random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                   random_state=SEED),
}
res = {}
naive = np.maximum(X_all.loc[te, "prepared_qty"] - X_all.loc[te, "roll7_sold"], 0)
res["Naive (prepared - 7-day avg)"] = metrics(y[te], naive)
preds = {}
for name, m in models.items():
    m.fit(X_all.loc[tr, waste_feats], y[tr])
    p = np.maximum(m.predict(X_all.loc[te, waste_feats]), 0)
    preds[name] = p; res[name] = metrics(y[te], p)
print("\nWaste prediction on the test set (servings):")
for k, v in res.items():
    print(f"  {k:30s} MAE={v['MAE']:.2f}  RMSE={v['RMSE']:.2f}  R2={v['R2']:.3f}")
best = min(models, key=lambda k: res[k]["MAE"])
print("Best waste model:", best)
imp = pd.Series(models["Random Forest"].feature_importances_, index=waste_feats).sort_values(ascending=False)

# ---------- Task 2: demand prediction -> recommended preparation ----------
yd = X_all["demand_qty"]
dm = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
dm.fit(X_all.loc[tr, demand_feats], yd[tr])
pd_te = dm.predict(X_all.loc[te, demand_feats])
dres = metrics(yd[te], pd_te)
print(f"\nDemand model (Random Forest): MAE={dres['MAE']:.2f} RMSE={dres['RMSE']:.2f} R2={dres['R2']:.3f}")

T = df.loc[te].copy(); T["pred_demand"] = pd_te
def evaluate(rec):
    waste = np.maximum(rec - T.demand_qty, 0); short = np.maximum(T.demand_qty - rec, 0)
    return dict(prepared=int(rec.sum()), waste=int(waste.sum()), waste_pct=waste.sum() / rec.sum() * 100,
                shortage=int(short.sum()), fill=(1 - short.sum() / T.demand_qty.sum()) * 100,
                waste_cost=float((waste * T.unit_cost).sum()))
policy = {"Current practice": evaluate(T.prepared_qty)}
for b in (0.0, 0.05, 0.10, 0.15):
    rec = np.ceil(T.pred_demand * (1 + b))
    policy[f"ML + {int(b*100)}% buffer"] = evaluate(rec)
print("\nPreparation policies on the test period:")
for k, v in policy.items():
    print(f"  {k:18s} prepared={v['prepared']:6d} waste={v['waste']:5d} ({v['waste_pct']:.1f}%) "
          f"shortage={v['shortage']:4d} fill={v['fill']:.1f}% waste_cost=Rs {v['waste_cost']:.0f}")
CHOSEN = "ML + 10% buffer"
base_w = policy["Current practice"]["waste"]; ch = policy[CHOSEN]
red = (base_w - ch["waste"]) / base_w * 100
red_cost = (policy["Current practice"]["waste_cost"] - ch["waste_cost"]) / policy["Current practice"]["waste_cost"] * 100
print(f"\nWith {CHOSEN}: waste reduced by {red:.1f}% and waste cost by {red_cost:.1f}%")

# per item
T["rec"] = np.ceil(T.pred_demand * 1.10)
T["w_base"] = T.waste_qty; T["w_ml"] = np.maximum(T.rec - T.demand_qty, 0)
per_item = T.groupby("item")[["w_base", "w_ml"]].sum()
per_item["reduction_pct"] = (per_item.w_base - per_item.w_ml) / per_item.w_base * 100
print("\nPer-item waste (test period):"); print(per_item.round(1).to_string())

# ---------- save artefacts ----------
joblib.dump(dict(waste_model=models[best], demand_model=dm, waste_feats=waste_feats, demand_feats=demand_feats,
                 items=items, best=best, buffer=0.10), "models/food_waste_model.joblib", compress=3)
out = dict(rows=len(df), train=int(tr.sum()), test=int(te.sum()), cut=str(cut.date()), waste_models=res, best=best,
           demand=dres, policy=policy, chosen=CHOSEN, reduction=red, reduction_cost=red_cost,
           per_item=per_item.round(2).reset_index().to_dict("records"),
           importance=imp.head(10).round(4).to_dict(),
           overall_waste_pct=float(df.waste_qty.sum() / df.prepared_qty.sum() * 100),
           total_waste=int(df.waste_qty.sum()), total_prepared=int(df.prepared_qty.sum()))
json.dump(out, open("models/metrics.json", "w"), indent=1)

# ---------- figures ----------
sns.set_theme(style="whitegrid", font="DejaVu Serif", font_scale=1.0)
# EDA 1: heatmap day-of-week x item (mean waste %)
hm = df.pivot_table(index="item", columns="day_of_week", values="waste_pct", aggfunc="mean")
hm.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(7.5, 4.2)); sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Mean waste (% of prepared)"})
plt.ylabel(""); plt.xlabel("Day of week"); plt.tight_layout(); plt.savefig("figures/eda_heatmap.png", dpi=200); plt.close()
# EDA 2: monthly waste trend
mt = df.groupby(df.date.dt.to_period("M"))[["waste_qty"]].sum(); mt.index = mt.index.to_timestamp()
plt.figure(figsize=(7.5, 3.6)); plt.plot(mt.index, mt.waste_qty, marker="o", color="#c0392b")
plt.ylabel("Servings wasted per month"); plt.xlabel("Month"); plt.tight_layout(); plt.savefig("figures/eda_monthly.png", dpi=200); plt.close()
# Model comparison
names = [k for k in res]; mae = [res[k]["MAE"] for k in names]; r2 = [res[k]["R2"] for k in names]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
cols = ["#95a5a6", "#3498db", "#27ae60", "#e67e22"]
lab = ["Naive", "Linear\nRegression", "Random\nForest", "Gradient\nBoosting"]
ax[0].bar(lab, mae, color=cols); ax[0].set_title("Mean absolute error (lower is better)"); ax[0].set_ylabel("Servings")
ax[1].bar(lab, r2, color=cols); ax[1].set_title("R² score (higher is better)")
for a, vals in zip(ax, (mae, r2)):
    for i, v in enumerate(vals): a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig("figures/model_comparison.png", dpi=200); plt.close()
# Actual vs predicted
bp = preds[best]
plt.figure(figsize=(5.4, 5)); plt.scatter(y[te], bp, s=8, alpha=0.45, color="#2c7fb8")
mx = max(y[te].max(), bp.max()); plt.plot([0, mx], [0, mx], "r--", lw=1.2, label="Perfect prediction")
plt.xlabel("Actual waste (servings)"); plt.ylabel("Predicted waste (servings)"); plt.title(f"{best}: actual vs predicted")
plt.legend(); plt.tight_layout(); plt.savefig("figures/actual_vs_pred.png", dpi=200); plt.close()
# Feature importance
top = imp.head(10)[::-1]
plt.figure(figsize=(7, 4)); plt.barh(top.index, top.values, color="#16a085"); plt.xlabel("Importance (Random Forest)")
plt.tight_layout(); plt.savefig("figures/feature_importance.png", dpi=200); plt.close()
# Per item baseline vs ML
pi = per_item.sort_values("w_base", ascending=False); x = np.arange(len(pi))
plt.figure(figsize=(8, 4)); plt.bar(x - 0.2, pi.w_base, 0.4, label="Current practice", color="#e74c3c")
plt.bar(x + 0.2, pi.w_ml, 0.4, label=CHOSEN, color="#27ae60"); plt.xticks(x, pi.index, rotation=25, ha="right")
plt.ylabel("Servings wasted (test period)"); plt.legend(); plt.tight_layout(); plt.savefig("figures/waste_by_item.png", dpi=200); plt.close()
# Trade-off curve
bs = [0, 5, 10, 15]; keys = list(policy)[1:]
fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.plot(bs, [policy[k]["waste"] for k in keys], "o-", color="#e67e22")
ax.set_xlabel("Safety buffer added to predicted demand (%)"); ax.set_ylabel("Servings wasted", color="#e67e22")
ax2 = ax.twinx(); ax2.plot(bs, [policy[k]["fill"] for k in keys], "s--", color="#2980b9"); ax2.set_ylabel("Demand fulfilled (%)", color="#2980b9")
ax2.grid(False); plt.tight_layout(); plt.savefig("figures/tradeoff.png", dpi=200); plt.close()
print(f"\nTotal run time: {time.time()-t0:.1f} s")
