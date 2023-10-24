from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from enum import Enum
import re
from typing import Any


class ParquetLogicalTypes(str, Enum):
    STRING = "STRING"
    ENUM = "ENUM"
    UUID = "UUID"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    DECIMAL = "DECIMAL"
    DATE = "DATE"
    TIME_MILLIS = "TIME_MILLIS"
    TIME_MICROS = "TIME_MICROS"
    TIME_NANOS = "TIME_NANOS"
    TIMESTAMP_MILLIS = "TIMESTAMP_MILLIS"
    TIMESTAMP_MICROS = "TIMESTAMP_MICROS"
    TIMESTAMP_NANOS = "TIMESTAMP_NANOS"
    INTERVAL = "INTERVAL"
    JSON = "JSON"
    BSON = "BSON"


class Entity(BaseModel):
    name: str
    description: str
    keys: list[str]


class FeatureStore(BaseModel):
    entities: list[Entity]
    time_key: str
    feature_sets: list[FeatureSet]


class FeatureSet(BaseModel):
    """This class represents the metadata for a table store within the Feature Store."""

    name: str
    description: str
    mnemonic: str
    tags: list[str] = Field(default=[])
    entity: Entity
    features: list[Feature]


class Feature(BaseModel):
    """This class represents the metadata for a field within a table."""

    name: str
    description: str
    type: ParquetLogicalTypes
    tags: list[str] = Field(default=[])
    domain: list[Domain] = Field(default=[])


class RangeDomain(BaseModel):
    """This class represents the metadata for a range domain of a feature"""

    type: Literal["RANGE"] = "RANGE"
    start: float | int
    include_start: bool = Field(default=True)
    end: float | int
    include_end: bool = Field(default=True)
    description: str

    def __init__(self, **data):
        (
            data["start"],
            data["include_start"],
            data["end"],
            data["include_end"],
        ) = self.from_str(data["value"])
        super().__init__(**data)

    @staticmethod
    def from_str(value: str) -> tuple[float, bool, float, bool]:
        pattern = r"^(\[|\()([^,]+),([^,]+)(\]|\))$"
        match = re.match(pattern, value)
        if not match:
            raise ValueError("Invalid range interval format")

        include_start = match.group(1) == "["
        include_end = match.group(4) == "]"

        try:
            start_val = float(match.group(2).strip())
            end_val = float(match.group(3).strip())

        except ValueError:
            raise ValueError("Invalid number format in the interval")

        if start_val > end_val:
            raise ValueError("Start value must be less than or equal to end value")
        return start_val, include_start, end_val, include_end


class ValueDomain(BaseModel):
    """This class represents the metadata for a value domain of a feature"""

    type: Literal["VALUE"] = "VALUE"
    value: float | int
    description: str


class TextDomain(BaseModel):
    """This class represents the metadata for a text domain of a feature"""

    type: Literal["TEXT"] = "TEXT"
    value: str
    description: str


Domain = Annotated[RangeDomain | ValueDomain | TextDomain, Field(discriminator="type")]
