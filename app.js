const ids = ["item","date","temperature","rain","is_holiday","is_event","expected_footfall","planned_qty"];
async function loadHistory() {
  const rows = await (await fetch("/api/history")).json();
  const tb = document.querySelector("#hist tbody"); tb.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");
    [r.item_name, r.target_date, r.planned_qty, r.predicted_waste.toFixed(1), r.recommended_qty, r.risk_level]
      .forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
    tb.appendChild(tr);
  });
}
document.getElementById("go").addEventListener("click", async () => {
  const body = {}; ids.forEach(i => body[i] = document.getElementById(i).value);
  ["temperature","expected_footfall","planned_qty"].forEach(k => body[k] = Number(body[k]));
  ["rain","is_holiday","is_event"].forEach(k => body[k] = Number(body[k]));
  const err = document.getElementById("error"); err.textContent = "";
  const resp = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  const data = await resp.json();
  if (!resp.ok) { err.textContent = data.error; return; }
  document.getElementById("result").hidden = false;
  document.getElementById("r_waste").textContent = data.predicted_waste + " (" + data.waste_percent + "%)";
  document.getElementById("r_demand").textContent = data.predicted_demand;
  document.getElementById("r_rec").textContent = data.recommended_qty;
  const rk = document.getElementById("r_risk"); rk.textContent = data.risk; rk.className = "risk-" + data.risk;
  document.getElementById("r_msg").textContent = data.saving > 0
    ? "Preparing " + data.recommended_qty + " instead of " + data.planned_qty + " servings avoids about " + data.saving + " servings of waste (Rs " + data.cost_saving + ")."
    : "The planned quantity is already at or below the recommended level.";
  loadHistory();
});
loadHistory();
