from pathlib import Path

import pandas as pd
import structlog

from src.utils.config import settings

logger = structlog.get_logger()


class DataCleaner:
    def __init__(self) -> None:
        self.interim_dir = settings.data_dir / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def _read_file(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        else:
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.read_csv(path)

    def clean_portwatch(self, path: Path) -> pd.DataFrame:
        df = self._read_file(path)
        df["daily_port_calls"] = df["daily_port_calls"].clip(lower=0)
        df["incoming_shipment_volume_mt"] = df["incoming_shipment_volume_mt"].clip(lower=0).fillna(0)
        df["outgoing_shipment_volume_mt"] = df["outgoing_shipment_volume_mt"].clip(lower=0).fillna(0)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def clean_chokepoints(self, path: Path) -> pd.DataFrame:
        df = self._read_file(path)
        df["daily_transit_calls"] = df["daily_transit_calls"].clip(lower=0)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def clean_lsci(self, path: Path) -> pd.DataFrame:
        df = self._read_file(path)
        df["date"] = pd.to_datetime(df["date"])
        if "country_code" in df.columns:
            df = df.sort_values(["country_code", "date"]).reset_index(drop=True)
            df = df.groupby("country_code").ffill(limit=3).bfill(limit=1)
        return df

    def clean_lpi(self, path: Path) -> pd.DataFrame:
        df = self._read_file(path)
        return df

    def clean_gdacs(self, path: Path) -> pd.DataFrame:
        df = self._read_file(path)
        df["date"] = pd.to_datetime(df["date"])
        if "event_id" in df.columns:
            df = df.drop_duplicates(subset=["event_id"])
        if "alert_level" in df.columns:
            alert_map = {"green": 1, "orange": 2, "red": 3}
            df["alert_level_num"] = df["alert_level"].map(alert_map).fillna(0)
        else:
            df["alert_level_num"] = 0
        return df

    def process_all(self, paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
        logger.info("Cleaning data")
        return {
            "portwatch": self.clean_portwatch(paths["portwatch"]),
            "chokepoints": self.clean_chokepoints(paths["chokepoints"]),
            "lsci": self.clean_lsci(paths["lsci"]),
            "lpi": self.clean_lpi(paths["lpi"]),
            "gdacs": self.clean_gdacs(paths["gdacs"]),
        }


if __name__ == "__main__":
    from src.data.ingestion import DataIngestor

    ingestor = DataIngestor()
    paths = ingestor.ingest_all()
    cleaner = DataCleaner()
    clean_data = cleaner.process_all(paths)
    for k, df in clean_data.items():
        df.to_parquet(cleaner.interim_dir / f"{k}_clean.parquet")
    logger.info("Cleaning completed")
