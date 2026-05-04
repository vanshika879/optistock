import argparse
import logging
import json
import os
from datetime import datetime

from optistock.data_loader import DataLoader
from optistock.demand_correction import LatentDemandCorrector
from optistock.forecaster import DemandForecaster
from optistock.optimizer import InventoryOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(product_id: int, horizon: int, sample_size: int, output_dir: str):
    logger.info("=" * 60)
    logger.info(f"Starting OptiStock Pipeline for Product {product_id}")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    loader = DataLoader()
    df_hourly = loader.load_and_expand_data(sample_size=sample_size)
    
    # 2. Correct Latent Demand
    corrector = LatentDemandCorrector()
    df_corrected = corrector.correct_demand(df_hourly)
    
    # 3. Forecast
    forecaster = DemandForecaster(horizon_days=horizon)
    forecast_period = forecaster.forecast(df_corrected, product_id)
    
    # 4. Optimize Inventory
    optimizer = InventoryOptimizer(n_trials=50) # Use 50 for faster CLI runs
    policy, df_sim, summary = optimizer.optimize(forecast_period)
    
    # 5. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Summary JSON
    summary_path = os.path.join(output_dir, f"summary_{product_id}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "product_id": product_id,
            "horizon_days": horizon,
            "optimal_policy": policy,
            "performance": summary
        }, f, indent=4)
        
    # Save Simulation CSV
    sim_path = os.path.join(output_dir, f"simulation_{product_id}_{timestamp}.csv")
    df_sim.to_csv(sim_path, index=False)
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="OptiStock: Demand Forecasting & Inventory Optimization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    run_parser = subparsers.add_parser("run", help="Run the full optimization pipeline")
    run_parser.add_argument("--product", type=int, default=267, help="Product ID to analyze")
    run_parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    run_parser.add_argument("--sample", type=int, default=70000, help="Number of records to load from dataset")
    run_parser.add_argument("--output", type=str, default="output", help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_pipeline(args.product, args.horizon, args.sample, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
