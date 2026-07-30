import time
from pathlib import Path

import requests
import structlog
from bs4 import BeautifulSoup

from src.utils.config import settings

logger = structlog.get_logger()

class DataIngestor:
    def __init__(self) -> None:
        self.raw_dir = settings.data_dir / "raw"
        self.samples_dir = settings.data_dir / "samples"
        for subdir in ["portwatch", "chokepoints", "lsci", "lpi", "gdacs"]:
            (self.raw_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _fallback_to_sample(self, name: str) -> Path:
        logger.warning(f"Falling back to sample data for {name}")
        sample_path = self.samples_dir / f"{name}_sample.parquet"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample data not found: {sample_path}")
        return sample_path

    def _retry_get(self, url: str, retries: int = 3, **kwargs) -> requests.Response:
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=10, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2 ** attempt)
        raise Exception(f"Failed to fetch {url} after {retries} attempts.")

    def download_portwatch(self) -> Path:
        """Downloads IMF PortWatch data."""
        if settings.use_sample_data:
            return self._fallback_to_sample("portwatch")
        try:
            res = self._retry_get("https://portwatch.imf.org/search?collection=dataset")
            soup = BeautifulSoup(res.text, 'html.parser')
            links = soup.find_all('a', href=lambda href: href and '.csv' in href)
            if not links:
                raise Exception("No CSV links found on PortWatch.")

            csv_url = "https://portwatch.imf.org" + str(links[0]['href'])
            csv_res = self._retry_get(csv_url)

            out_path = self.raw_dir / "portwatch" / "portwatch_raw.csv"
            with open(out_path, "wb") as f:
                f.write(csv_res.content)
            return out_path
        except Exception as e:
            logger.error("Portwatch download failed", error=str(e))
            return self._fallback_to_sample("portwatch")

    def download_chokepoints(self) -> Path:
        if settings.use_sample_data:
            return self._fallback_to_sample("chokepoints")
        try:
            res = self._retry_get("https://portwatch.imf.org/search?collection=dataset")
            soup = BeautifulSoup(res.text, 'html.parser')
            links = soup.find_all('a', href=lambda href: href and 'chokepoint' in href.lower() and '.csv' in href)
            if not links:
                raise Exception("No chokepoint CSV links found.")

            csv_url = "https://portwatch.imf.org" + str(links[0]['href'])
            csv_res = self._retry_get(csv_url)

            out_path = self.raw_dir / "chokepoints" / "chokepoints_raw.csv"
            with open(out_path, "wb") as f:
                f.write(csv_res.content)
            return out_path
        except Exception as e:
            logger.error("Chokepoints download failed", error=str(e))
            return self._fallback_to_sample("chokepoints")

    def download_lsci(self) -> Path:
        if settings.use_sample_data:
            return self._fallback_to_sample("lsci")
        try:
            res = self._retry_get("https://unctadstat.unctad.org/datacentre/api/v1/US.LSCI/csv")
            out_path = self.raw_dir / "lsci" / "lsci_raw.csv"
            with open(out_path, "wb") as f:
                f.write(res.content)
            return out_path
        except Exception as e:
            logger.error("LSCI download failed", error=str(e))
            return self._fallback_to_sample("lsci")

    def download_lpi(self) -> Path:
        if settings.use_sample_data:
            return self._fallback_to_sample("lpi")
        try:
            res = self._retry_get("https://data360.worldbank.org/en/api/WB_LPI_20/csv")
            out_path = self.raw_dir / "lpi" / "lpi_raw.csv"
            with open(out_path, "wb") as f:
                f.write(res.content)
            return out_path
        except Exception as e:
            logger.error("LPI download failed", error=str(e))
            return self._fallback_to_sample("lpi")

    def download_gdacs(self) -> Path:
        if settings.use_sample_data:
            return self._fallback_to_sample("gdacs")
        try:
            response = self._retry_get(settings.gdacs_api_url)
            out_path = self.raw_dir / "gdacs" / "gdacs_raw.json"
            with open(out_path, "wb") as f:
                f.write(response.content)
            return out_path
        except Exception as e:
            logger.error("GDACS download failed", error=str(e))
            return self._fallback_to_sample("gdacs")

    def ingest_all(self) -> dict[str, Path]:
        return {
            "portwatch": self.download_portwatch(),
            "chokepoints": self.download_chokepoints(),
            "lsci": self.download_lsci(),
            "lpi": self.download_lpi(),
            "gdacs": self.download_gdacs()
        }
