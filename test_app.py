"""Functional test cases for the web application and model (results are printed as a table)."""
import os, json, time, joblib
os.environ["FOODWASTE_DB"] = "test_foodwaste.db"
if os.path.exists("test_foodwaste.db"): os.remove("test_foodwaste.db")
import database; database.init_db()
from app import app
import pandas as pd

c = app.test_client(); results = []
def check(tid, desc, cond, actual):
    results.append((tid, desc, "Pass" if cond else "FAIL", actual)); print(f"{tid}: {desc:55s} -> {actual} [{'Pass' if cond else 'FAIL'}]")

good = dict(item="Veg Biryani", date="2026-01-05", temperature=27, rain=0, is_holiday=0, is_event=0,
            expected_footfall=650, planned_qty=220)
con = database.connect()
n_items = con.execute("SELECT COUNT(*) FROM food_items").fetchone()[0]; n_sales = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]; con.close()
check("TC1", "Database created and loaded from dataset", n_items == 8 and n_sales == 5848, f"{n_items} items, {n_sales} sales rows")
r = c.get("/"); check("TC2", "Home page loads", r.status_code == 200 and b"Food Waste Prediction" in r.data, f"HTTP {r.status_code}")
r = c.post("/api/predict", json=good); d = r.get_json()
check("TC3", "Valid request returns a prediction", r.status_code == 200 and d["predicted_waste"] >= 0, f"HTTP {r.status_code}, waste={d.get('predicted_waste')}")
check("TC4", "Recommended quantity is a positive integer", isinstance(d["recommended_qty"], int) and d["recommended_qty"] > 0, f"{d['recommended_qty']} servings")
check("TC5", "Risk level is Low, Medium or High", d["risk"] in ("Low", "Medium", "High"), d["risk"])
bad = dict(good); del bad["rain"]; r = c.post("/api/predict", json=bad)
check("TC6", "Missing field is rejected", r.status_code == 400, f"HTTP {r.status_code}")
bad = dict(good, item="Pizza"); r = c.post("/api/predict", json=bad)
check("TC7", "Unknown food item is rejected", r.status_code == 400, f"HTTP {r.status_code}")
bad = dict(good, planned_qty=-5); r = c.post("/api/predict", json=bad)
check("TC8", "Negative quantity is rejected", r.status_code == 400, f"HTTP {r.status_code}")
r = c.get("/api/history"); h = r.get_json()
check("TC9", "Prediction is stored and listed in history", r.status_code == 200 and len(h) == 1, f"{len(h)} record stored")
hi = c.post("/api/predict", json=dict(good, planned_qty=400)).get_json(); lo = c.post("/api/predict", json=dict(good, planned_qty=170)).get_json()
check("TC10", "Larger planned quantity gives higher predicted waste", hi["predicted_waste"] > lo["predicted_waste"], f"{hi['predicted_waste']} vs {lo['predicted_waste']}")
t = time.time(); c.post("/api/predict", json=good); ms = (time.time() - t) * 1000
check("TC11", "Prediction response time below 1 second", ms < 1000, f"{ms:.0f} ms")
m = json.load(open("models/metrics.json")); b = m["waste_models"][m["best"]]
check("TC12", "Waste model R2 on unseen test period at least 0.60", b["R2"] >= 0.60, f"R2 = {b['R2']:.3f}")
ch = m["policy"][m["chosen"]]; base = m["policy"]["Current practice"]
check("TC13", "Model-based plan wastes less than current practice", ch["waste"] < base["waste"], f"{ch['waste']} vs {base['waste']} servings")
check("TC14", "Demand fulfilled stays at or above 95%", ch["fill"] >= 95, f"{ch['fill']:.1f}%")
print(f"\n{sum(1 for x in results if x[2]=='Pass')} of {len(results)} test cases passed")
json.dump(results, open("figures/test_results.json", "w"))
os.remove("test_foodwaste.db")
