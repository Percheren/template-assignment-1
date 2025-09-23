from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
import xarray as xr
from gurobipy import GRB
from src.data_ops import data_loader


class OptModel:
    """
    Placeholder for optimization models using Gurobipy.

    Attributes (examples):
        N (int): Number of time steps/consumers/etc.
        question/scenario name (str): Configuration/question identifier.
        ...
    """
    
    
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
    