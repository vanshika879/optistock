import ast
import logging
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data ingestion and preprocessing for the OptiStock pipeline."""
    
    def __init__(self, dataset_name: str = "Dingdong-Inc/FreshRetailNet-50K"):
        self.dataset_name = dataset_name

    def load_and_expand_data(self, sample_size: int = 70000) -> pd.DataFrame:
        """
        Loads the dataset from Hugging Face and expands daily summary records 
        into hourly timeseries records.
        """
        logger.info(f"Loading dataset '{self.dataset_name}' from Hugging Face...")
        
        try:
            dataset = load_dataset(self.dataset_name)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        df_train_full = dataset["train"].to_pandas()
        df_train = df_train_full.tail(sample_size).copy()
        df_train.reset_index(drop=True, inplace=True)
        
        logger.info(f"Loaded {len(df_train):,} base records. Parsing arrays...")

        def safe_eval(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return x
        
        df_train["hours_sale"] = df_train["hours_sale"].apply(safe_eval)
        df_train["hours_stock_status"] = df_train["hours_stock_status"].apply(safe_eval)
        
        logger.info("Expanding daily summaries into hourly records...")
        
        rows = []
        for _, row in df_train.iterrows():
            # Only process valid rows where lists have 24 hours
            if len(row["hours_sale"]) < 24 or len(row["hours_stock_status"]) < 24:
                continue
                
            for hour in range(24):
                rows.append({
                    "city_id": row["city_id"],
                    "store_id": row["store_id"],
                    "product_id": row["product_id"],
                    "dt": row["dt"],
                    "hour": hour,
                    "sale": float(row["hours_sale"][hour]),
                    "stock_status": int(row["hours_stock_status"][hour]),
                    "stock_hour6_22_cnt": row["stock_hour6_22_cnt"],
                    "discount": row["discount"],
                    "holiday_flag": row["holiday_flag"],
                    "activity_flag": row["activity_flag"],
                    "precpt": row["precpt"],
                    "avg_temperature": row["avg_temperature"],
                    "avg_humidity": row["avg_humidity"],
                    "avg_wind_level": row["avg_wind_level"]
                })
        
        df_hourly = pd.DataFrame(rows)
        logger.info(f"Expansion complete: {len(df_hourly):,} hourly records generated.")
        
        return df_hourly
