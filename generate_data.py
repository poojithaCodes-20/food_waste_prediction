"""Generate a reproducible synthetic food-service dataset (2 years, 8 menu items)."""
import numpy as np, pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

ITEMS = {  # name: (share of footfall, unit cost INR, selling price INR)
    "Veg Biryani":     (0.20, 45, 90),
    "Chicken Biryani": (0.15, 80, 150),
    "Idli":            (0.22, 12, 30),
    "Dosa":            (0.18, 20, 50),
    "Chapati":         (0.30, 6, 15),
    "Curd Rice":       (0.12, 18, 45),
    "Fried Rice":      (0.14, 35, 80),
    "Samosa":          (0.17, 8, 20),
}
HOLIDAYS = set(pd.to_datetime([
    "2024-01-15","2024-01-26","2024-03-25","2024-04-09","2024-04-11","2024-08-15",
    "2024-10-02","2024-10-12","2024-11-01","2024-11-15","2024-12-25",
    "2025-01-14","2025-01-26","2025-03-14","2025-03-31","2025-04-14","2025-08-15",
    "2025-10-02","2025-10-20","2025-11-05","2025-12-25"]))

def main():
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n = len(dates)
    dow = dates.dayofweek.values
    month = dates.month.values
    is_weekend = (dow >= 5).astype(int)
    is_holiday = np.array([d in HOLIDAYS for d in dates]).astype(int)
    event_days = rng.choice(n, size=int(n * 0.07), replace=False)
    is_event = np.zeros(n, int); is_event[event_days] = 1
    # temperature (deg C) with seasonal cycle and noise; rain probability by season
    temp = 29 + 7 * np.sin(2 * np.pi * (dates.dayofyear.values - 90) / 365) + rng.normal(0, 1.8, n)
    rain_p = 0.05 + 0.35 * ((month >= 6) & (month <= 9))
    rain = (rng.random(n) < rain_p).astype(int)
    dow_f = np.array([1.00, 1.00, 1.02, 1.03, 1.08, 1.18, 1.12])[dow]
    footfall = 600 * dow_f * np.where(is_holiday == 1, 0.55, 1.0) * np.where(is_event == 1, 1.35, 1.0) \
               * np.where(rain == 1, 0.90, 1.0) * np.exp(rng.normal(0, 0.05, n))
    expected_footfall = footfall * np.exp(rng.normal(0, 0.06, n))  # forecast known in advance

    rows = []
    for item, (share, cost, price) in ITEMS.items():
        hot = (temp - 29) / 7.0
        if "Curd" in item: tf = 1 + 0.18 * hot
        elif "Biryani" in item: tf = 1 - 0.07 * hot
        elif item in ("Idli", "Dosa"): tf = 1 + 0.04 * (rain == 1)
        else: tf = np.ones(n)
        wk = 1 + (0.10 if item == "Chicken Biryani" else 0.0) * is_weekend
        demand = footfall * share * tf * wk * np.exp(rng.normal(0, 0.09, n))
        demand = np.round(demand).astype(int)
        prepared = np.zeros(n, int); sold = np.zeros(n, int)
        hist = []
        for i in range(n):
            base = np.mean(hist[-7:]) if len(hist) >= 7 else demand[i]
            prepared[i] = int(round(base * rng.uniform(1.12, 1.35)))
            sold[i] = min(prepared[i], demand[i])
            hist.append(sold[i])
        waste = np.maximum(prepared - demand, 0)
        df = pd.DataFrame(dict(date=dates, item=item, day_of_week=dow, month=month,
                               is_weekend=is_weekend, is_holiday=is_holiday, is_event=is_event,
                               temperature=temp.round(1), rain=rain,
                               expected_footfall=expected_footfall.round(0).astype(int),
                               unit_cost=cost, price=price, prepared_qty=prepared,
                               sold_qty=sold, demand_qty=demand, waste_qty=waste))
        rows.append(df)
    data = pd.concat(rows).sort_values(["item", "date"]).reset_index(drop=True)
    data.to_csv("data/food_waste.csv", index=False)
    print("rows:", len(data), "columns:", data.shape[1])

if __name__ == "__main__":
    main()
