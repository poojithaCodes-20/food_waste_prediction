"""Flask web application that serves the trained model and stores records in the database."""
import datetime as dt
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request
import database

app = Flask(__name__)
BUNDLE = joblib.load("models/food_waste_model.joblib")
REQUIRED = ["item", "date", "temperature", "rain", "is_holiday", "is_event", "expected_footfall", "planned_qty"]

def risk_level(pct):
    return "Low" if pct < 10 else ("Medium" if pct < 20 else "High")

def build_features(f, item_row, hist):
    d = dt.date.fromisoformat(f["date"])
    sold = hist["sold_qty"].tolist(); waste = hist["waste_qty"].tolist()
    row = dict(day_of_week=d.weekday(), month=d.month, is_weekend=int(d.weekday() >= 5),
               is_holiday=int(f["is_holiday"]), is_event=int(f["is_event"]), temperature=float(f["temperature"]),
               rain=int(f["rain"]), expected_footfall=float(f["expected_footfall"]),
               unit_cost=item_row["unit_cost"], price=item_row["price"], prepared_qty=float(f["planned_qty"]),
               lag1_sold=sold[-1], lag1_waste=waste[-1], roll7_sold=float(np.mean(sold[-7:])),
               roll7_waste=float(np.mean(waste[-7:])))
    for it in BUNDLE["items"]:
        row["item_" + it] = int(it == f["item"])
    return pd.DataFrame([row])

@app.route("/")
def index():
    return render_template("index.html", items=BUNDLE["items"])

@app.route("/api/predict", methods=["POST"])
def predict():
    f = request.get_json(silent=True) or {}
    missing = [k for k in REQUIRED if k not in f]
    if missing:
        return jsonify(error="Missing fields: " + ", ".join(missing)), 400
    if f["item"] not in BUNDLE["items"]:
        return jsonify(error="Unknown food item"), 400
    try:
        dt.date.fromisoformat(f["date"])
        if float(f["planned_qty"]) <= 0 or float(f["expected_footfall"]) <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify(error="Invalid date or non-positive quantity"), 400
    con = database.connect()
    item_row = con.execute("SELECT * FROM food_items WHERE item_name=?", (f["item"],)).fetchone()
    hist = pd.read_sql_query("SELECT sold_qty, waste_qty FROM sales WHERE item_id=? AND sale_date<? "
                             "ORDER BY sale_date DESC LIMIT 7", con, params=(int(item_row["item_id"]), f["date"]))
    hist = hist.iloc[::-1]
    if hist.empty:
        con.close(); return jsonify(error="No sales history available for this item"), 400
    X = build_features(f, item_row, hist)
    waste = float(max(BUNDLE["waste_model"].predict(X[BUNDLE["waste_feats"]])[0], 0))
    demand = float(BUNDLE["demand_model"].predict(X[BUNDLE["demand_feats"]])[0])
    rec = int(np.ceil(demand * (1 + BUNDLE["buffer"])))
    planned = int(float(f["planned_qty"]))
    pct = waste / planned * 100
    out = dict(item=f["item"], date=f["date"], planned_qty=planned, predicted_waste=round(waste, 1),
               waste_percent=round(pct, 1), risk=risk_level(pct), predicted_demand=round(demand, 1),
               recommended_qty=rec, saving=max(planned - rec, 0),
               cost_saving=round(max(planned - rec, 0) * item_row["unit_cost"], 2))
    con.execute("INSERT INTO predictions(item_id,target_date,planned_qty,predicted_waste,predicted_demand,"
                "recommended_qty,risk_level) VALUES (?,?,?,?,?,?,?)",
                (int(item_row["item_id"]), f["date"], planned, waste, demand, rec, out["risk"]))
    con.commit(); con.close()
    return jsonify(out)

@app.route("/api/history")
def history():
    con = database.connect()
    rows = con.execute("SELECT p.prediction_id, i.item_name, p.target_date, p.planned_qty, p.predicted_waste, "
                       "p.recommended_qty, p.risk_level FROM predictions p JOIN food_items i USING(item_id) "
                       "ORDER BY p.prediction_id DESC LIMIT 10").fetchall()
    con.close(); return jsonify([dict(r) for r in rows])

@app.route("/api/summary")
def summary():
    con = database.connect()
    r = con.execute("SELECT SUM(prepared_qty) p, SUM(sold_qty) s, SUM(waste_qty) w FROM sales").fetchone()
    con.close()
    return jsonify(prepared=r["p"], sold=r["s"], wasted=r["w"], waste_percent=round(r["w"] / r["p"] * 100, 1))

if __name__ == "__main__":
    database.init_db(); app.run(debug=False)
