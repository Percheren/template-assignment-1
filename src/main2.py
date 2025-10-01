import json
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB


# Function to load JSON data
def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def run_flex_load_pv_optimization(load_params, bus_params, der_production, usage_preference):
    
    # Define model parameters
    T = len(der_production[0]["hourly_profile_ratio"])
    pv_max = der_production[0]["hourly_profile_ratio"]
    min_daily_energy = usage_preference[0]["load_preferences"][0]["min_total_energy_per_day_hour_equivalent"]
    max_import = bus_params[0]["max_import_kW"]
    max_export = bus_params[0]["max_export_kW"]
    import_tariff = bus_params[0]["import_tariff_DKK/kWh"]
    export_tariff = bus_params[0]["export_tariff_DKK/kWh"]
    energy_price = bus_params[0]["energy_price_DKK_per_kWh"]

    max_load = load_params["FFL_01"]["max_load_kWh_per_hour"]


    # Create optimization model
    m = gp.Model("flexible_load_pv")


    # Decision variables
    PD = {t: m.addVar(lb=0, ub=max_load, name=f"PD_{t}") for t in range(T)}
    PG = {t: m.addVar(lb=0, ub=pv_max[t], name=f"PG_{t}") for t in range(T)}
    PIMP = {t: m.addVar(lb=0, ub=max_import, name=f"PIMP_{t}") for t in range(T)}
    PEXP = {t: m.addVar(lb=0, ub=max_export, name=f"PEXP_{t}") for t in range(T)}
    PV_curtail = {t: m.addVar(lb=0, ub=pv_max[t], name=f"PV_curtail_{t}") for t in range(T)}



    # Constraints
    for t in range(T):
        m.addConstr(PIMP[t] - PEXP[t] == PD[t] - PG[t], name=f"balance_{t}")
        m.addConstr(PG[t] + PV_curtail[t] == pv_max[t], name=f"pv_cap_{t}")

    m.addConstr(gp.quicksum(PD[t] for t in range(T)) >= min_daily_energy, name="daily_min_energy")


    # Full objective function
    obj = gp.quicksum(
        energy_price[t] * (PIMP[t] - PEXP[t]) +
        import_tariff * PIMP[t] +
        export_tariff * PEXP[t] for t in range(T)
    )
    
    # Set objective 
    m.setObjective(obj, GRB.MINIMIZE)

    # Optimize model
    m.optimize()
    
    # Check if optimization is successful
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Solver ended with status {m.Status}")


    # Extract results
    results = []
    for t in range(T):
        imp = PIMP[t].X
        exp = PEXP[t].X
        pv_used = PG[t].X
        demand = PD[t].X
        net_grid = imp - exp
        pv_curt = PV_curtail[t].X
        results.append([t, imp, exp, pv_used, demand, net_grid, pv_curt])

    return results, m.ObjVal


def main():
    # File paths
    base_dir = Path(__file__).resolve().parent.parent / "data" / "question_1a"

    # Load input data
    appliance_params = load_json(base_dir / "appliance_params.json")
    bus_params = load_json(base_dir / "bus_params.json")
    der_production = load_json(base_dir / "DER_production.json")
    usage_preference = load_json(base_dir / "usage_preference.json")
    load_params = {load["load_id"]: load for load in appliance_params["load"]}
    
    # Change in input parameters for a5
    
    

    # Run optimization
    results, obj_val = run_flex_load_pv_optimization(load_params, bus_params, der_production, usage_preference)

    resultsa5, obj_vala5 = run_flex_load_pv_optimization(load_params, bus_params, der_production, usage_preference)
    # Print results
    print(f"Objective (total cost): {obj_val:.2f} DKK\n")
    print("Hour | Import | Export | PV_used | Demand | NetGrid | PV_Curtail")
    print("-" * 70)
    for row in results:
        print(f"{row[0]:>4} | {row[1]:>6.2f} | {row[2]:>6.2f} | {row[3]:>7.2f} | {row[4]:>6.2f} | {row[5]:>7.2f} | {row[6]:>9.2f}")
    #print("Total PV-production:",sum(results[row][3] for row in range(len(results))))
    #print("Total consumption:", round(sum(results[row][4] for row in range(len(results)))))
if __name__ == "__main__":
    main()
    
    
    