import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class LatentDemandCorrector:
    """
    Corrects censored demand (when sales drop to zero because stock is zero)
    using a LightGBM regressor to predict what the true demand would have been.
    """
    
    def __init__(self, learning_rate: float = 0.05, num_leaves: int = 31):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def correct_demand(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Takes hourly data and adds an 'imputed_sale' column where stockout periods
        are replaced with model-predicted latent demand.
        """
        logger.info("Starting latent demand correction...")
        
        df = df_hourly.copy()
        df["dt"] = pd.to_datetime(df["dt"])
        df["weekday"] = df["dt"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
        df = df.sort_values(["product_id", "store_id", "dt", "hour"])
        
        # Feature Engineering: Lags and Rolling Means
        df["sale_shift1"] = df.groupby(["product_id", "store_id"])["sale"].shift(1).fillna(0)
        df["sale_roll24"] = (
            df.groupby(["product_id", "store_id"])["sale"]
            .rolling(window=24, min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
        ).fillna(0)
        
        features = [
            "hour", "weekday", "is_weekend", "discount", "holiday_flag",
            "avg_temperature", "sale_shift1", "sale_roll24"
        ]
        
        # We train the model ONLY on hours where the product was IN STOCK
        train_df = df[df["stock_status"] == 0].copy()
        
        if len(train_df) < 50:
            logger.warning("Insufficient training data for LightGBM. Using mean imputation.")
            df["imputed_sale"] = np.where(df["stock_status"] == 0, df["sale"], df["sale"].mean())
            return df
            
        X = train_df[features].fillna(0)
        y = train_df["sale"].values
        
        test_size = min(0.2, max(0.1, 10/len(X)))
        
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
        except ValueError as e:
            logger.warning(f"Train/test split failed: {e}. Using mean imputation.")
            df["imputed_sale"] = np.where(df["stock_status"] == 0, df["sale"], df["sale"].mean())
            return df
            
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "verbose": -1,
            "force_col_wise": True
        }
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        logger.info("Training LightGBM model for latent demand prediction...")
        try:
            model = lgb.train(
                params, 
                lgb_train,
                valid_sets=[lgb_train, lgb_val],
                num_boost_round=min(1000, len(X_tr) * 2),
                callbacks=callbacks
            )
            
            df["imputed_sale"] = df["sale"].copy()
            out_of_stock_mask = df["stock_status"] == 1
            
            if out_of_stock_mask.sum() > 0:
                X_pred = df.loc[out_of_stock_mask, features].fillna(0)
                predictions = model.predict(X_pred)
                # Cap negative predictions to 0
                predictions = np.maximum(predictions, 0)
                df.loc[out_of_stock_mask, "imputed_sale"] = predictions
                logger.info(f"Successfully imputed {out_of_stock_mask.sum():,} stockout hours.")
            
            return df
            
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}. Using simple mean imputation.")
            mean_sale = train_df["sale"].mean()
            df["imputed_sale"] = np.where(df["stock_status"] == 0, df["sale"], mean_sale)
            return df
