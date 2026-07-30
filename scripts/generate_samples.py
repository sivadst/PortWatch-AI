import pandas as pd
import numpy as np
from pathlib import Path
import random

def generate_samples():
    """Generates synthetic data with realistic patterns and saves to data/samples."""
    np.random.seed(42)
    random.seed(42)
    
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
    
    # 1. Port Activity
    ports = [f"PORT{str(i).zfill(3)}" for i in range(1, 51)]
    countries = [f"C{str(i).zfill(2)}" for i in range(1, 11)]
    port_data = []
    
    for port in ports:
        country = random.choice(countries)
        baseline = np.random.lognormal(mean=3, sigma=1)
        
        for date in dates:
            seasonality = np.sin(2 * np.pi * date.dayofyear / 365) * (baseline * 0.2)
            weekly = np.sin(2 * np.pi * date.dayofweek / 7) * (baseline * 0.1)
            
            disruption = 0
            if "2020-03" in str(date) or "2021-08" in str(date):
                disruption = - (baseline * 0.4)
                
            calls = max(0, int(baseline + seasonality + weekly + disruption + np.random.normal(0, baseline*0.1)))
            in_vol = calls * np.random.uniform(1000, 5000)
            out_vol = calls * np.random.uniform(800, 4000)
            
            port_data.append({
                "date": date,
                "port_id": port,
                "port_name": f"Port of {port}",
                "country_code": country,
                "daily_port_calls": calls,
                "incoming_shipment_volume_mt": in_vol,
                "outgoing_shipment_volume_mt": out_vol
            })
            
    df_port = pd.DataFrame(port_data)
    df_port.to_parquet(samples_dir / "portwatch_sample.parquet", index=False)
    
    # 2. Chokepoints
    chokepoints = ["Suez", "Panama", "Hormuz", "Malacca"]
    choke_data = []
    for cp in chokepoints:
        base = 50 if cp == "Suez" else 40
        for date in dates:
            drop = 0
            if cp == "Panama" and "2023" in str(date.year):
                drop = -15
            elif cp == "Suez" and "2021-03" in str(date)[:7]:
                drop = -40
            
            calls = max(0, int(base + drop + np.random.normal(0, 5)))
            choke_data.append({
                "date": date,
                "chokepoint_name": cp,
                "daily_transit_calls": calls,
                "transit_trade_volume_mt": calls * 2000
            })
    df_choke = pd.DataFrame(choke_data)
    df_choke.to_parquet(samples_dir / "chokepoints_sample.parquet", index=False)
    
    # 3. GDACS Disasters
    disasters = []
    for _ in range(500):
        date = np.random.choice(dates)
        disasters.append({
            "event_id": f"EQ{np.random.randint(100000)}",
            "event_type": random.choice(["EQ", "TC", "FL"]),
            "event_name": f"Disaster {_}",
            "country": random.choice(countries),
            "alert_level": random.choice(["red", "orange", "green"]),
            "date": date,
            "latitude": np.random.uniform(-90, 90),
            "longitude": np.random.uniform(-180, 180),
            "population_affected": np.random.randint(0, 1000000)
        })
    df_gdacs = pd.DataFrame(disasters)
    df_gdacs.to_parquet(samples_dir / "gdacs_sample.parquet", index=False)
    
    # 4. LSCI and LPI
    lsci_data = []
    lpi_data = []
    for c in countries:
        base_lsci = np.random.uniform(20, 150)
        base_lpi = np.random.uniform(2.0, 4.5)
        for year in range(2019, 2025):
            for month in range(1, 13):
                lsci_data.append({
                    "country_code": c,
                    "date": pd.Timestamp(f"{year}-{month:02d}-01"),
                    "lsci_value": max(0, base_lsci + np.random.normal(0, 5)),
                    "num_services": int(base_lsci),
                    "num_companies": int(base_lsci / 2),
                    "max_vessel_size": 15000,
                    "num_ships": int(base_lsci * 5),
                    "container_capacity": int(base_lsci * 10000)
                })
            
            lpi_data.append({
                "country_code": c,
                "year": year,
                "lpi_score": max(1.0, min(5.0, base_lpi + np.random.normal(0, 0.1))),
                "maritime_dwell_time_days": 10.0 / base_lpi,
                "customs_score": base_lpi * 0.9,
                "infrastructure_score": base_lpi * 0.95
            })
            
    pd.DataFrame(lsci_data).to_parquet(samples_dir / "lsci_sample.parquet", index=False)
    pd.DataFrame(lpi_data).to_parquet(samples_dir / "lpi_sample.parquet", index=False)
    
    print("Synthetic sample data generated successfully in data/samples/")

if __name__ == "__main__":
    generate_samples()
