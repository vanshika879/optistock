# OptiStock 📈📦

**OptiStock** is an intelligent demand forecasting and stochastic inventory optimization pipeline. It solves a classic supply chain problem: predicting actual demand when historical sales data is censored by stockouts, and determining exactly when and how much to reorder to maximize profit.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 The Problem Solved

Traditional forecasting models train on historical sales. But what happens if a product was out of stock for 3 days? The sales data reads `0`, but the actual *demand* was not zero. Training a model on this "censored" data leads to massive under-forecasting and perpetual out-of-stock loops.

**OptiStock solves this by:**
1. **Latent Demand Correction:** Uses **LightGBM** to predict what sales *would have been* during stockout periods based on weather, pricing, and historical lags.
2. **Robust Forecasting:** Uses **Facebook Prophet** to forecast the corrected demand into the future.
3. **Stochastic Optimization:** Uses **Optuna** and Monte Carlo simulation to find the exact Reorder Point (RP) and Reorder Quantity (RQ) that minimizes holding, stockout, and perishability costs while maintaining a 93%+ Service Level.

## 🏗 Architecture

The codebase is strictly modularized for enterprise scalability:

```text
optistock/
├── data_loader.py         # Ingests & expands HuggingFace datasets (FreshRetailNet)
├── demand_correction.py   # LightGBM Latent Demand Imputation
├── forecaster.py          # Prophet Time-Series Forecasting
├── optimizer.py           # Optuna Stochastic Inventory Simulation
└── cli.py                 # Command-line entry point
```

## 🚀 Quick Start

1. **Clone & Install**
```bash
git clone https://github.com/yourusername/optistock.git
cd optistock
pip install -r requirements.txt
```

2. **Run the Pipeline**
Run an end-to-end optimization for a specific product (e.g., Product 267) over a 30-day forecast horizon.
```bash
python -m optistock.cli run --product 267 --horizon 30
```

3. **Check the Output**
The pipeline will generate a JSON summary containing the optimal policy and a CSV of the daily inventory simulation in the `output/` directory.

## 📊 Sample Output (JSON)

```json
{
    "product_id": 267,
    "horizon_days": 30,
    "optimal_policy": {
        "reorder_point": 142,
        "reorder_quantity": 380,
        "lead_time": 2,
        "holding_cost_per_unit": 1.0,
        "stockout_cost_per_unit": 4.0,
        "waste_cost_per_unit": 2.5,
        "perish_days": 3
    },
    "performance": {
        "total_cost": 4250.50,
        "service_level": 0.945,
        "waste_rate": 0.08,
        "avg_end_inventory": 210.4
    }
}
```

## 🛠 Tech Stack
* **Pandas / NumPy**: Data wrangling and vectorization
* **LightGBM**: Gradient boosting for latent demand imputation
* **Prophet**: Additive regression model for time-series forecasting
* **Optuna**: Hyperparameter optimization framework for policy search
* **Hugging Face Datasets**: Open-source data ingestion

---
*Built for modern supply chain analytics.*
