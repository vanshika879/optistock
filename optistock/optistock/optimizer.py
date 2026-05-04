import logging
import numpy as np
import pandas as pd
import optuna
from typing import Tuple, Dict

# Disable optuna print spam
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

class InventoryOptimizer:
    """
    Simulates inventory levels against stochastic demand and uses Optuna
    to find the optimal reorder point and reorder quantity.
    """
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials

    def simulate(self, forecast_df: pd.DataFrame, policy: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate inventory dynamics with perishability over the forecasted period.
        """
        np.random.seed(42) # For reproducible stochastic demand
        
        rp = policy["reorder_point"]
        rq = policy["reorder_quantity"]
        lt = policy["lead_time"]
        hc = policy["holding_cost_per_unit"]
        sc = policy["stockout_cost_per_unit"]
        wc = policy["waste_cost_per_unit"]
        init_inv = policy["initial_inventory"]
        perish_days = policy.get("perish_days", 3)
        
        # inv tracks lists of [quantity, age_in_days]
        inv = [[init_inv, 0]]
        pending_orders = []
        results = []
        
        for i, row in forecast_df.reset_index(drop=True).iterrows():
            # Stochastic demand: actual demand fluctuates around forecast
            std = max(1e-3, row["forecast_demand"] * 0.1)
            actual_demand = max(0, np.random.normal(row["forecast_demand"], std))
            
            # 1. Process arriving orders
            arrived = [o for o in pending_orders if o[0] == i]
            for arrival in arrived:
                inv.append([arrival[1], 0])
                pending_orders.remove(arrival)
            
            # 2. Serve demand (FIFO based on age)
            total_inv = sum(q for q, _ in inv)
            sales = min(total_inv, actual_demand)
            lost_sales = actual_demand - sales
            
            to_sell = sales
            new_inv = []
            for qty, age in inv:
                if to_sell <= 0:
                    new_inv.append([qty, age])
                elif qty <= to_sell:
                    to_sell -= qty
                else:
                    new_inv.append([qty - to_sell, age])
                    to_sell = 0
            inv = new_inv
            
            # 3. Age inventory and process spoilage
            new_inv = []
            waste_today = 0
            for qty, age in inv:
                age2 = age + 1
                if age2 >= perish_days:
                    waste_today += qty
                else:
                    new_inv.append([qty, age2])
            inv = new_inv
            
            # 4. Calculate daily costs
            end_inv = sum(q for q, _ in inv)
            holding_cost = end_inv * hc
            stockout_cost = lost_sales * sc
            waste_cost = waste_today * wc
            total_cost = holding_cost + stockout_cost + waste_cost
            
            # 5. Reorder logic
            inventory_position = end_inv + sum(q for _, q in pending_orders)
            if inventory_position < rp:
                pending_orders.append((i + lt, rq))
            
            results.append({
                "date": row["date"],
                "forecast_demand": row["forecast_demand"],
                "actual_demand": actual_demand,
                "sales": sales,
                "lost_sales": lost_sales,
                "waste": waste_today,
                "end_inventory": end_inv,
                "total_cost": total_cost
            })
        
        df_sim = pd.DataFrame(results)
        
        fulfilled = df_sim["sales"].sum()
        total_demand = df_sim["actual_demand"].sum()
        total_waste = df_sim["waste"].sum()
        
        summary = {
            "total_cost": float(df_sim["total_cost"].sum()),
            "total_lost_sales": float(df_sim["lost_sales"].sum()),
            "total_waste": float(total_waste),
            "service_level": float(fulfilled / total_demand) if total_demand > 0 else 1.0,
            "waste_rate": float(total_waste / (fulfilled + total_waste)) if (fulfilled + total_waste) > 0 else 0.0,
            "avg_end_inventory": float(df_sim["end_inventory"].mean())
        }
        
        return df_sim, summary

    def optimize(self, forecast_period: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, Dict]:
        """
        Uses Optuna to find the RP and RQ that minimizes total costs while 
        maintaining target service levels.
        """
        logger.info(f"Running Optuna optimization ({self.n_trials} trials)...")
        
        def objective(trial):
            # Hyperparameters to tune
            reorder_point = trial.suggest_int("reorder_point", 50, 500)
            reorder_quantity = trial.suggest_int("reorder_quantity", 100, 600)
            
            policy = {
                "reorder_point": reorder_point,
                "reorder_quantity": reorder_quantity,
                "lead_time": 2,
                "holding_cost_per_unit": 1.0,
                "stockout_cost_per_unit": 4.0,
                "waste_cost_per_unit": 2.5,
                "initial_inventory": 300,
                "perish_days": 3
            }
            
            _, summary = self.simulate(forecast_period, policy)
            
            total_cost = summary["total_cost"]
            service_level = summary["service_level"]
            
            # Penalize policies that don't meet target service levels
            if service_level < 0.93:
                penalty = (0.93 - service_level) * 1000000
            elif service_level > 0.98:
                penalty = (service_level - 0.98) * 5000  # Slight penalty for over-servicing (too much stock)
            else:
                penalty = 0
                
            return total_cost + penalty
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        optimal_policy = {
            **study.best_trial.params,
            "lead_time": 2,
            "holding_cost_per_unit": 1.0,
            "stockout_cost_per_unit": 4.0,
            "waste_cost_per_unit": 2.5,
            "initial_inventory": 300,
            "perish_days": 3
        }
        
        # Run final simulation with the winning policy
        df_sim, summary_best = self.simulate(forecast_period, optimal_policy)
        
        logger.info(f"Optimization complete. Best RP: {optimal_policy['reorder_point']}, Best RQ: {optimal_policy['reorder_quantity']}")
        logger.info(f"Achieved Service Level: {summary_best['service_level']:.2%}, Total Cost: ${summary_best['total_cost']:,.2f}")
        
        return optimal_policy, df_sim, summary_best
