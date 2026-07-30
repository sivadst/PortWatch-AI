import sqlite3
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:
    duckdb = None

import pandas as pd
import structlog

from src.utils.config import settings

logger = structlog.get_logger()


class DataService:
    def __init__(self) -> None:
        self.duck_db_path = str(settings.data_dir / "processed" / "portwatch.duckdb")
        self.sqlite_db_path = str(settings.data_dir / "processed" / "portwatch.db")
        self.parquet_path = settings.data_dir / "processed" / "features.parquet"

    def get_port_features(self, port_id: str) -> dict[str, Any]:
        """Fetch the most recent features for a given port_id with resilient fallbacks."""
        # 1. Try DuckDB
        if duckdb is not None and Path(self.duck_db_path).exists():
            try:
                with duckdb.connect(self.duck_db_path) as con:
                    df = con.execute(
                        "SELECT * FROM port_activity WHERE port_id = ? ORDER BY date DESC LIMIT 1",
                        [port_id],
                    ).df()
                    if not df.empty:
                        return self._format_feature_record(df.iloc[0].to_dict())
            except Exception as e:
                logger.warning(f"DuckDB query failed, attempting SQLite fallback: {e}")

        # 2. Try SQLite
        if Path(self.sqlite_db_path).exists():
            try:
                conn = sqlite3.connect(self.sqlite_db_path)
                df = pd.read_sql_query(
                    "SELECT * FROM port_activity WHERE port_id = ? ORDER BY date DESC LIMIT 1",
                    conn,
                    params=[port_id],
                )
                conn.close()
                if not df.empty:
                    return self._format_feature_record(df.iloc[0].to_dict())
            except Exception as e:
                logger.warning(f"SQLite query failed, attempting Parquet fallback: {e}")

        # 3. Try Parquet
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
                port_df = df[df["port_id"] == port_id].sort_values("date", ascending=False)
                if not port_df.empty:
                    return self._format_feature_record(port_df.iloc[0].to_dict())
            except Exception as e:
                logger.warning(f"Parquet query failed: {e}")

        return {}

    def _format_feature_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "port_calls_lag_1d": float(record.get("port_calls_lag_1d", record.get("daily_port_calls", 0))),
            "port_calls_lag_2d": float(record.get("port_calls_lag_2d", record.get("daily_port_calls", 0))),
            "port_calls_lag_7d": float(record.get("port_calls_lag_7d", record.get("daily_port_calls", 0))),
            "global_chokepoint_transit": float(record.get("global_chokepoint_transit", 0)),
            "active_disasters_7d": float(record.get("active_disasters_7d", 0)),
            "congestion_index": float(record.get("congestion_index", 0.0)),
            "congestion_level": str(record.get("congestion_level", "NORMAL")),
        }

    def get_ports(self) -> list[dict[str, Any]]:
        """Fetch list of monitored ports."""
        if Path(self.sqlite_db_path).exists():
            try:
                conn = sqlite3.connect(self.sqlite_db_path)
                df = pd.read_sql_query("SELECT DISTINCT port_id, port_name, country_code FROM ports LIMIT 100", conn)
                conn.close()
                return df.to_dict(orient="records")
            except Exception:
                pass

        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
                ports_df = df[["port_id", "port_name", "country_code"]].drop_duplicates().head(100)
                return ports_df.to_dict(orient="records")
            except Exception:
                pass

        return []


data_service = DataService()
