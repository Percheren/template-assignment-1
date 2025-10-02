import json
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

# -------------------------------
# Question 1.a implementation
# -------------------------------
def run_flex_load_pv_optimization(load_params1a, bus_params1a, der_production1a, usage_preference1a,appliance_params1a):
    T = len(der_production1a[0]["hourly_profile_ratio"])
    pv_max = der_production1a[0]["hourly_profile_ratio"]
    min_daily_energy = usage_preference1a[0]["load_preferences"][0]["min_total_energy_per_day_hour_equivalent"]

    cap = appliance_params1a["DER"][0]["max_power_kW"]
    max_import = bus_params1a[0]["max_import_kW"]
    max_export = bus_params1a[0]["max_export_kW"]
    import_tariff = bus_params1a[0]["import_tariff_DKK/kWh"]
    export_tariff = bus_params1a[0]["export_tariff_DKK/kWh"]
    energy_price = bus_params1a[0]["energy_price_DKK_per_kWh"]

    max_load = load_params1a["FFL_01"]["max_load_kWh_per_hour"]

    m = gp.Model("flexible_load_pv_1a")

    PD = {t: m.addVar(lb=0, ub=max_load, name=f"PD_{t}") for t in range(T)}
    PG = {t: m.addVar(lb=0, ub=pv_max[t]*cap, name=f"PG_{t}") for t in range(T)}
    PIMP = {t: m.addVar(lb=0, ub=max_import, name=f"PIMP_{t}") for t in range(T)}
    PEXP = {t: m.addVar(lb=0, ub=max_export, name=f"PEXP_{t}") for t in range(T)}
    PV_curtail = {t: m.addVar(lb=0, ub=cap, name=f"PV_curtail_{t}") for t in range(T)}

    # Constraints
    for t in range(T):
        m.addConstr(PG[t] + PV_curtail[t] == pv_max[t] * cap, name=f"generation_{t}")
        m.addConstr(PIMP[t] - PEXP[t] == PD[t] - PG[t], name=f"balance_{t}")
        #m.addConstr((PG[t]-PD[t])-(PEXP[t]-PIMP[t]) == PV_curtail[t], name=f"pv_cap_{t}")

    m.addConstr(gp.quicksum(PD[t] for t in range(T)) >= min_daily_energy, name="daily_min_energy")

    # Objective
    obj = gp.quicksum(
        energy_price[t] * (PIMP[t] - PEXP[t]) +
        import_tariff * PIMP[t] +
        export_tariff * PEXP[t] for t in range(T)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver ended with status {m.Status}")

    results = []
    for t in range(T):
        results.append([
            t,
            PIMP[t].X,
            PEXP[t].X,
            PG[t].X,
            PD[t].X,
            PIMP[t].X - PEXP[t].X,
            PV_curtail[t].X
        ])

    return results, m.ObjVal

# -------------------------------
# Question 1.b implementation
# -------------------------------
def run_flex_load_with_discomfort(load_params1b, bus_params1b, der_production1b, usage_preference1b, appliance_params1b):
    """
    1.b optimization: minimize energy cost + discomfort penalty from deviating
    from a reference hourly profile.
    """
    T = len(der_production1b[0]["hourly_profile_ratio"])
    pv_max = der_production1b[0]["hourly_profile_ratio"]
    cap = appliance_params1b["DER"][0]["max_power_kW"]

    load_pref = usage_preference1b[0]["load_preferences"][0]

    # Reference profile handling
    if load_pref.get("hourly_profile_ratio") is not None:
        ref_profile = load_pref["hourly_profile_ratio"]
    else:
        min_daily_energy = load_pref["min_total_energy_per_day_hour_equivalent"]
        ref_profile = [min_daily_energy / T] * T
        print("⚠️ No reference profile in JSON, using flat profile instead.")

    ref_profile = [x * cap for x in ref_profile]

    #total_ref_energy = sum(ref_profile)

    
    max_import = bus_params1b[0]["max_import_kW"]
    max_export = bus_params1b[0]["max_export_kW"]
    import_tariff = bus_params1b[0]["import_tariff_DKK/kWh"]
    export_tariff = bus_params1b[0]["export_tariff_DKK/kWh"]
    energy_price = bus_params1b[0]["energy_price_DKK_per_kWh"]

    max_load = load_params1b["FFL_01"]["max_load_kWh_per_hour"]

    m = gp.Model("flexible_load_pv_1b")

    PD_flex = {t: m.addVar(lb=0, ub=max_load, name=f"PDflex_{t}") for t in range(T)}
    PG = {t: m.addVar(lb=0, ub=pv_max[t]*cap, name=f"PG_{t}") for t in range(T)}
    PIMP = {t: m.addVar(lb=0, ub=max_import, name=f"PIMP_{t}") for t in range(T)}
    PEXP = {t: m.addVar(lb=0, ub=max_export, name=f"PEXP_{t}") for t in range(T)}
    PV_curtail = {t: m.addVar(lb=0, ub=cap, name=f"PV_curtail_{t}") for t in range(T)}
    U = {t: m.addVar(lb=0, name=f"U_{t}") for t in range(T)}  # absolute deviation vars

    # Constraints
    for t in range(T):
        m.addConstr(PIMP[t] - PEXP[t] == PD_flex[t] - PG[t], name=f"balance_{t}")
        m.addConstr(PG[t] + PV_curtail[t] == pv_max[t] * cap, name=f"generation_{t}")
        #m.addConstr(PG[t] + PV_curtail[t] == pv_max[t], name=f"pv_cap_{t}")
        m.addConstr(U[t] >= PD_flex[t] - ref_profile[t], name=f"dev_pos_{t}")
        m.addConstr(U[t] >= -(PD_flex[t] - ref_profile[t]), name=f"dev_neg_{t}")

    # Conserve daily flexible energy equal to reference total
    #m.addConstr(gp.quicksum(PD_flex[t] for t in range(T)) == total_ref_energy, name="daily_energy_match")

    # Objective: cost + discomfort
    obj = gp.quicksum(
        energy_price[t] * (PIMP[t] - PEXP[t]) +
        import_tariff * PIMP[t] +
        export_tariff * PEXP[t] +
        U[t]  for t in range(T)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver ended with status {m.Status}")

    results = []
    for t in range(T):
        results.append([
            t,
            PIMP[t].X,
            PEXP[t].X,
            PG[t].X,
            PD_flex[t].X,
            U[t].X,
            PIMP[t].X - PEXP[t].X,
            PV_curtail[t].X
        ])

    return results, m.ObjVal

# -------------------------------
# Example main to run both
# -------------------------------
def main():
    base_dir1a = Path(__file__).resolve().parent.parent / "data" / "question_1a"
    base_dir1b = Path(__file__).resolve().parent.parent / "data" / "question_1b"

    appliance_params1a = load_json(base_dir1a / "appliance_params.json")
    bus_params1a = load_json(base_dir1a / "bus_params.json")
    der_production1a = load_json(base_dir1a / "DER_production.json")
    usage_preference1a = load_json(base_dir1a / "usage_preference.json")
    load_params1a = {load["load_id"]: load for load in appliance_params1a["load"]}

    appliance_params1b = load_json(base_dir1b / "appliance_params.json")
    bus_params1b = load_json(base_dir1b / "bus_params.json")
    der_production1b = load_json(base_dir1b / "DER_production.json")
    usage_preference1b = load_json(base_dir1b / "usage_preferences.json")
    load_params1b = {load["load_id"]: load for load in appliance_params1b["load"]}

    # Run 1.a
    results_a, obj_val_a = run_flex_load_pv_optimization(load_params1a, bus_params1a, der_production1a, usage_preference1a, appliance_params1a)
    print(f"1.a Objective (total cost): {obj_val_a:.2f} DKK\n")
    print("Hour | Import | Export | PV_used | Demand | NetGrid | PV_Curtail")
    print("-" * 70)
    for row in results_a:
        print(f"{row[0]:>4} | {row[1]:>6.2f} | {row[2]:>6.2f} | {row[3]:>7.2f} | {row[4]:>6.2f} | {row[5]:>7.2f} | {row[6]:>9.2f}")

    # Run 1.b
    results_b, obj_val_b = run_flex_load_with_discomfort(load_params1b, bus_params1b, der_production1b, usage_preference1b, appliance_params1b)
    print(f"\n1.b Objective (cost + discomfort): {obj_val_b:.3f} DKK\n")
    print("Hour | Import | Export | PV_used | Demand | Deviation | NetGrid | PV_Curtail")
    print("-" * 90)
    for row in results_b:
        print(f"{row[0]:>4} | {row[1]:>6.3f} | {row[2]:>6.3f} | {row[3]:>7.3f} | {row[4]:>6.3f} | {row[5]:>9.3f} | {row[6]:>7.3f} | {row[7]:>9.3f}")

if __name__ == "__main__":
    main()