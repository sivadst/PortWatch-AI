import pandas as pd

from src.data.cleaning import DataCleaner
from src.features.build_features import assign_level


def test_clean_portwatch(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "daily_port_calls": [-5, 10],  # negative should be clipped
        "incoming_shipment_volume_mt": [100, None],
        "outgoing_shipment_volume_mt": [None, 200],
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)

    cleaner = DataCleaner()
    cleaned = cleaner.clean_portwatch(path)

    assert cleaned["daily_port_calls"].iloc[0] == 0  # clipped
    assert cleaned["daily_port_calls"].iloc[1] == 10
    assert cleaned["incoming_shipment_volume_mt"].iloc[1] == 0  # fillna 0
    assert cleaned["outgoing_shipment_volume_mt"].iloc[0] == 0


def test_assign_level():
    assert assign_level(-1.0) == "LOW"
    assert assign_level(0.0) == "NORMAL"
    assert assign_level(1.0) == "HIGH"
    assert assign_level(2.0) == "CRITICAL"
