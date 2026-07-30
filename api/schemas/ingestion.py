from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PortActivityRecord(BaseModel):
    date: date
    port_id: str
    port_name: str
    country_code: str = Field(..., pattern=r"^[A-Z]{2,3}$")
    daily_port_calls: int = Field(..., ge=0)
    incoming_volume_mt: float | None = Field(None, ge=0)
    outgoing_volume_mt: float | None = Field(None, ge=0)

class ChokepointRecord(BaseModel):
    date: date
    chokepoint_name: str
    daily_transit_calls: int = Field(..., ge=0)
    transit_volume_mt: float | None = Field(None, ge=0)

class LSCIRecord(BaseModel):
    country_code: str = Field(..., pattern=r"^[A-Z]{2,3}$")
    date: date
    lsci_value: float = Field(..., ge=0)
    num_services: int | None = Field(None, ge=0)
    num_companies: int | None = Field(None, ge=0)

class LPI_Record(BaseModel):
    country_code: str = Field(..., pattern=r"^[A-Z]{2,3}$")
    year: int
    lpi_score: float

class GDACSRecord(BaseModel):
    event_id: str
    event_type: Literal["EQ", "TC", "FL", "VO", "DR"]
    country: str
    alert_level: Literal["red", "orange", "green"]
    date: date
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    population_affected: int | None = Field(None, ge=0)
