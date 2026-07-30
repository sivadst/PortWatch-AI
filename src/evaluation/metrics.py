from pathlib import Path

from src.models.train import ModelTrainer


def calculate_metrics() -> None:
    """Executes full evaluation pipeline across horizons and generates metrics reports."""
    trainer = ModelTrainer()
    trainer.run_pipeline()
    reports_dir = Path("reports")
    print(f"Metrics and benchmark results generated successfully in {reports_dir.absolute()}")


if __name__ == "__main__":
    calculate_metrics()
