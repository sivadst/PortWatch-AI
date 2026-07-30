import sqlite3

try:
    import duckdb
except ImportError:
    duckdb = None

import pandas as pd
import structlog

from src.utils.config import settings

logger = structlog.get_logger()


def assign_level(idx: float) -> str:
    """Classifies congestion index into discrete risk categories."""
    if pd.isna(idx):
        return "NORMAL"
    if idx < -0.5:
        return "LOW"
    if idx <= 0.5:
        return "NORMAL"
    if idx <= 1.5:
        return "HIGH"
    return "CRITICAL"


class FeatureEngineer:
    def __init__(self) -> None:
        self.interim_dir = settings.data_dir / "interim"
        self.processed_dir = settings.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.processed_dir / "portwatch.duckdb"
        self.sqlite_db_path = self.processed_dir / "portwatch.db"

    def build_features(self) -> pd.DataFrame:
        logger.info("Building features from interim data...")
        port_df = pd.read_parquet(self.interim_dir / "portwatch_clean.parquet")
        choke_df = pd.read_parquet(self.interim_dir / "chokepoints_clean.parquet")
        lsci_df = pd.read_parquet(self.interim_dir / "lsci_clean.parquet")
        gdacs_df = pd.read_parquet(self.interim_dir / "gdacs_clean.parquet")

        # 1. Merge Chokepoint daily aggregate transit
        choke_daily = choke_df.groupby("date")["daily_transit_calls"].sum().reset_index()
        choke_daily.rename(columns={"daily_transit_calls": "global_chokepoint_transit"}, inplace=True)
        df = pd.merge(port_df, choke_daily, on="date", how="left")
        df["global_chokepoint_transit"] = df["global_chokepoint_transit"].fillna(0)

        # 2. Merge LSCI
        lsci_df["year_month"] = lsci_df["date"].dt.to_period("M")
        df["year_month"] = df["date"].dt.to_period("M")
        if "country_code" in lsci_df.columns:
            lsci_agg = lsci_df.groupby(["country_code", "year_month"])["lsci_value"].mean().reset_index()
            df = pd.merge(df, lsci_agg, on=["country_code", "year_month"], how="left")
            df["lsci_value"] = df["lsci_value"].fillna(df["lsci_value"].median() if not df["lsci_value"].dropna().empty else 100.0)
        df.drop(columns=["year_month"], inplace=True)

        # 3. Merge GDACS Active Disasters (7-day window per country)
        gdacs_counts = gdacs_df.groupby(["country", "date"])["event_id"].count().reset_index()
        gdacs_counts.rename(columns={"event_id": "active_disasters_7d", "country": "country_code"}, inplace=True)
        df = pd.merge(df, gdacs_counts, on=["country_code", "date"], how="left")
        df["active_disasters_7d"] = df["active_disasters_7d"].fillna(0)

        # Ensure strict sorting by port and date
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["port_id", "date"]).reset_index(drop=True)

        # 4. Prevent Future Data Leakage for Historical Stats:
        # Calculate historical mean and std strictly up to train split date (or expanding window)
        train_cutoff = pd.to_datetime(settings.train_end_date)
        train_mask = df["date"] <= train_cutoff

        train_stats = (
            df[train_mask]
            .groupby("port_id")["daily_port_calls"]
            .agg(historical_mean="mean", historical_std="std")
            .reset_index()
        )
        # Global fallbacks if a port has zero train stats
        global_mean = df[train_mask]["daily_port_calls"].mean() if not df[train_mask].empty else df["daily_port_calls"].mean()
        global_std = df[train_mask]["daily_port_calls"].std() if not df[train_mask].empty else df["daily_port_calls"].std()

        df = pd.merge(df, train_stats, on="port_id", how="left")
        df["historical_mean"] = df["historical_mean"].fillna(global_mean)
        df["historical_std"] = df["historical_std"].fillna(global_std).replace(0, 1.0)

        # 5. Features at time t
        df["port_calls_7d_ma"] = df.groupby("port_id")["daily_port_calls"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df["congestion_index"] = (df["port_calls_7d_ma"] - df["historical_mean"]) / df["historical_std"]
        df["congestion_level"] = df["congestion_index"].apply(assign_level)

        # Lag features at t (strictly historical)
        for lag in [1, 2, 7]:
            df[f"port_calls_lag_{lag}d"] = df.groupby("port_id")["daily_port_calls"].shift(lag)

        # 6. Multi-Horizon Forecasting Target Generation (Prevent Future Leakage)
        # Shift congestion_index and congestion_level FORWARD to generate target at t+h
        for h in settings.supported_horizons:
            df[f"target_reg_{h}d"] = df.groupby("port_id")["congestion_index"].shift(-h)
            df[f"target_cls_{h}d"] = df.groupby("port_id")["congestion_level"].shift(-h)

        # Clean NaNs in basic lag features
        df = df.dropna(subset=["port_calls_lag_1d", "port_calls_lag_2d", "port_calls_lag_7d"]).reset_index(drop=True)

        # Save to parquet
        features_path = self.processed_dir / "features.parquet"
        df.to_parquet(features_path, index=False)
        logger.info(f"Features saved to {features_path} with shape {df.shape}")

        # Populate storage engines
        self.populate_databases(df)
        return df

    def populate_databases(self, df: pd.DataFrame) -> None:
        """Populates DuckDB and SQLite databases with fallback for restricted environments."""
        # SQLite export (always available native library)
        try:
            conn_sq = sqlite3.connect(str(self.sqlite_db_path))
            df.to_sql("port_activity", conn_sq, if_exists="replace", index=False)

            ports_df = df[["port_id", "port_name", "country_code"]].drop_duplicates()
            ports_df.to_sql("ports", conn_sq, if_exists="replace", index=False)
            conn_sq.close()
            logger.info("SQLite database populated successfully.")
        except Exception as e:
            logger.warning(f"Failed to populate SQLite: {e}")

        # DuckDB export
        if duckdb is not None:
            try:
                con = duckdb.connect(str(self.db_path))
                con.execute("CREATE OR REPLACE TABLE port_activity AS SELECT * FROM df")
                ports_df = df[["port_id", "port_name", "country_code"]].drop_duplicates()
                con.execute("CREATE OR REPLACE TABLE ports AS SELECT * FROM ports_df")
                con.close()
                logger.info("DuckDB database populated successfully.")
            except Exception as e:
                logger.warning(f"DuckDB population skipped/failed (environment fallback active): {e}")
        else:
            logger.info("DuckDB module unavailable; SQLite fallback active.")



if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.build_features()
