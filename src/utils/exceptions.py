class PortWatchError(Exception):
    """Base exception for PortWatch AI."""

class DataIngestionError(PortWatchError):
    """Raised when data ingestion fails."""

class ValidationError(PortWatchError):
    """Raised when data validation fails."""

class ModelNotFoundError(PortWatchError):
    """Raised when a required ML model is not found."""

class PredictionError(PortWatchError):
    """Raised when a prediction fails."""
