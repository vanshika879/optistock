import logging
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

class DemandForecaster:
    """
    Uses Facebook Prophet to forecast future demand based on the imputed 
    historical demand, taking into account weather, holidays, and discounts.
    """
    
    def __init__(self, horizon_days: int = 30):
        self.horizon_days = horizon_days

    def forecast(self, df_corrected: pd.DataFrame, product_id: int) -> pd.DataFrame:
        """
        Takes the hourly dataframe (with imputed_sale), aggregates to daily,
        trains Prophet, and returns a dataframe of future forecasted demand.
        """
        logger.info(f"Forecasting {self.horizon_days} days for product {product_id}...")
        
        # 1. Parse dates
        df = df_corrected.copy()
        df["dt"] = pd.to_datetime(df["dt"], errors='coerce')
        df = df.dropna(subset=["dt"])
        
        if len(df) == 0:
            raise ValueError("No valid dates in dataset")
            
        df["datetime"] = df["dt"] + pd.to_timedelta(df["hour"], unit="h")
        
        # 2. Filter for specific product
        df_p = df[df["product_id"] == product_id].copy()
        if len(df_p) == 0:
            raise ValueError(f"No data found for product_id={product_id}")
            
        df_p["date"] = df_p["datetime"].dt.date
        
        # 3. Aggregate to daily
        agg_rules = {
            "imputed_sale": "sum",
            "discount": "mean",
            "holiday_flag": "max",
            "precpt": "mean",
            "avg_temperature": "mean",
            "avg_humidity": "mean",
            "avg_wind_level": "mean"
        }
        
        # Only aggregate columns that actually exist
        valid_cols = [c for c in agg_rules.keys() if c in df_p.columns]
        agg_dict = {c: agg_rules[c] for c in valid_cols}
        
        df_daily = df_p.groupby("date", as_index=False).agg(agg_dict)
        df_daily = df_daily.rename(columns={"date": "ds", "imputed_sale": "y"})
        df_daily["ds"] = pd.to_datetime(df_daily["ds"])
        df_daily = df_daily.sort_values("ds")
        
        # 4. Fill missing days in the timeseries
        min_date = df_daily["ds"].min()
        max_date = df_daily["ds"].max()
        full_dates = pd.date_range(min_date, max_date, freq="D")
        
        df_daily = df_daily.set_index("ds").reindex(full_dates).rename_axis("ds").reset_index()
        df_daily[valid_cols] = df_daily[valid_cols].ffill().fillna(0)
        df_daily["y"] = df_daily["y"].clip(lower=0)
        
        # 5. Prophet Setup
        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1
        )
        
        regressors = ["holiday_flag", "discount", "avg_temperature", "avg_humidity", "avg_wind_level", "precpt"]
        active_regressors = [reg for reg in regressors if reg in df_daily.columns]
        
        for reg in active_regressors:
            m.add_regressor(reg)
            
        logger.info("Training Prophet model...")
        m.fit(df_daily[["ds", "y"] + active_regressors])
        
        # 6. Forecast
        future = m.make_future_dataframe(periods=self.horizon_days, freq="D")
        
        # Carry forward the last known values for regressors into the future
        last_vals = df_daily.iloc[-1]
        for reg in active_regressors:
            future[reg] = last_vals[reg]
            
        forecast = m.predict(future)
        
        # Extract just the future period
        forecast_period = forecast[["ds", "yhat"]].tail(self.horizon_days).rename(
            columns={"ds": "date", "yhat": "forecast_demand"}
        )
        forecast_period["forecast_demand"] = forecast_period["forecast_demand"].clip(lower=0)
        
        logger.info(f"Forecast complete. Generated {len(forecast_period)} days of predictions.")
        return forecast_period
