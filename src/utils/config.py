from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "PortWatch AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Data
    data_dir: Path = Path("data")
    use_sample_data: bool = True


    # Database
    database_url: str = "duckdb:///data/processed/portwatch.duckdb"

    # ML
    model_dir: Path = Path("models")
    random_seed: int = 42
    forecast_horizon: int = 7
    supported_horizons: list[int] = [1, 7, 14]

    # Temporal Validation Boundaries
    train_end_date: str = "2023-05-31"
    val_end_date: str = "2023-12-31"
    test_start_date: str = "2024-01-01"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]
    rate_limit: str = "100/minute"

    # External APIs
    gdacs_api_url: str = "https://www.gdacs.org/gdacsapi/api/Events/geteventlist/SEARCH"
    world_bank_api_url: str = "https://api.worldbank.org/v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

