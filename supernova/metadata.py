from __future__ import annotations
from pydantic import BaseModel, Field, model_validator, ValidationInfo, BeforeValidator
from typing import Literal, Annotated
from enum import Enum
import re
import pathlib
import yaml
from .query import Query
from random import choice, uniform
from pyspark.sql.types import (
    DataType,
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
    DateType,
    TimestampType,
    ByteType,
    ShortType,
)
from pyspark.sql import functions as F 
from pyspark.sql.column import Column
from functools import reduce
import warnings

class ParquetLogicalTypes(str, Enum):
    """This class represents the logical types of a parquet file"""

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
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"

    def is_int(self) -> bool:
        return self in [
            ParquetLogicalTypes.INT8,
            ParquetLogicalTypes.INT16,
            ParquetLogicalTypes.INT32,
            ParquetLogicalTypes.INT64,
            ParquetLogicalTypes.UINT8,
            ParquetLogicalTypes.UINT16,
            ParquetLogicalTypes.UINT32,
            ParquetLogicalTypes.UINT64,
        ]

    def is_float(self) -> bool:
        return self in [
            ParquetLogicalTypes.DECIMAL,
            ParquetLogicalTypes.TIME_MILLIS,
            ParquetLogicalTypes.TIME_MICROS,
            ParquetLogicalTypes.TIME_NANOS,
            ParquetLogicalTypes.TIMESTAMP_MILLIS,
            ParquetLogicalTypes.TIMESTAMP_MICROS,
            ParquetLogicalTypes.TIMESTAMP_NANOS,
            ParquetLogicalTypes.FLOAT,
            ParquetLogicalTypes.DOUBLE,
        ]

    def get_spark(self) -> DataType:
        return {
            ParquetLogicalTypes.STRING: StringType(),
            ParquetLogicalTypes.ENUM: StringType(),
            ParquetLogicalTypes.UUID: StringType(),
            ParquetLogicalTypes.INT8: ByteType(),
            ParquetLogicalTypes.INT16: ShortType(),
            ParquetLogicalTypes.INT32: IntegerType(),
            ParquetLogicalTypes.INT64: LongType(),
            ParquetLogicalTypes.UINT8: ByteType(),
            ParquetLogicalTypes.UINT16: ShortType(),
            ParquetLogicalTypes.UINT32: IntegerType(),
            ParquetLogicalTypes.UINT64: LongType(),
            ParquetLogicalTypes.DECIMAL: DecimalType(),  # Todo: Add precision and scale
            ParquetLogicalTypes.DATE: DateType(),
            ParquetLogicalTypes.TIME_MILLIS: IntegerType(),
            ParquetLogicalTypes.TIME_MICROS: LongType(),
            ParquetLogicalTypes.TIME_NANOS: LongType(),
            ParquetLogicalTypes.TIMESTAMP_MILLIS: TimestampType(),
            ParquetLogicalTypes.TIMESTAMP_MICROS: TimestampType(),
            ParquetLogicalTypes.TIMESTAMP_NANOS: TimestampType(),
            ParquetLogicalTypes.INTERVAL: StringType(),
            ParquetLogicalTypes.JSON: StringType(),
            ParquetLogicalTypes.BSON: StringType(),
            ParquetLogicalTypes.FLOAT: FloatType(),
            ParquetLogicalTypes.DOUBLE: DoubleType(),
        }[self]


class Entity(BaseModel):
    """This class represents the metadata for an entity within the Feature Store. An
    entity identifies what type of subject is represented by a row in an specific feature
    set. For example, if every row in a feature set contains data about a given customer
    this feature set should have an entity called "customer".

    Attributes:
        name: The name of the entity.
        description: A description of the entity.
        keys: A list of the names of the columns that uniquely identify the entity.
    """

    name: str
    description: str
    keys: list[str]


def set_feature_set_entity(feature_set: dict, info: ValidationInfo) -> FeatureSet:
    FeatureSet._available_entities = info.data["entities"]
    return FeatureSet(**feature_set)


class FeatureStore(BaseModel):
    """This calss represents a feature store. A feature store is a collection of tables
    that contain pre-calculated information related to a set of standardized entities,
    such as customers, products, etc.

    Attributes:
        entities: A list of entities that are supported by the feature store.
        time_key: The name of the column that contains the timestamp of the data.
        feature_sets: A list of feature sets that are available in the feature store.
    """

    entities: list[Entity]
    feature_sets: list[Annotated[FeatureSet, BeforeValidator(set_feature_set_entity)]]

    @staticmethod
    def from_folder(path: pathlib.Path | str) -> FeatureStore:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        store_file = path / "store.yml"
        with open(store_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        data["feature_sets"] = []
        for set_file in path.glob("feature_sets/*.yml"):
            with open(set_file, "r", encoding="utf-8") as file:
                set_data = yaml.safe_load(file)
                data["feature_sets"].append(set_data)

        return FeatureStore(**data)

    def query(self, query: list[str]) -> Query:
        return Query(self, query)

    def get_feature_set(self, mnemonic: str) -> FeatureSet:
        feature_set = next(
            filter(lambda s: s.mnemonic == mnemonic, self.feature_sets), None
        )
        if feature_set is None:
            raise ValueError(
                f"Feature set {mnemonic} not found in the store. "
                f"Available feature sets: {[s.mnemonic for s in self.feature_sets]}"
            )
        return feature_set

    @staticmethod
    def read_yaml(path: pathlib.Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)


class FeatureSet(BaseModel):
    """This class represents the metadata for a table store within the Feature Store."""

    name: str
    description: str
    path: str
    mnemonic: str
    tags: list[str] = Field(default=[])
    entity: Entity
    date_field: str
    features: list[Feature]

    _available_entities: list[Entity]

    def __hash__(self):
        return hash(self.mnemonic)

    @model_validator(mode="before")
    def set_entity(cls, values: dict):
        entity = next(
            filter(lambda e: e.name == values["entity"], cls._available_entities), None
        )
        if entity is None:
            raise ValueError(
                f"Entity {values['entity']} not found in the store. "
                f"Available entities: {[e.name for e in cls._available_entities]}"
            )
        values["entity"] = entity
        return values

    def generate_row(self) -> dict[str, float | int | str | None]:
        return {f.name: f.generate() for f in self.features}

    def generate(self, n: int) -> list[dict[str, float | int | str | None]]:
        return [self.generate_row() for _ in range(n)]


class Feature(BaseModel):
    """This class represents the metadata for a field within a table."""

    name: str
    description: str
    type: ParquetLogicalTypes
    tags: list[str] = Field(default=[])
    domain: list[Domain] = Field(default=[])

    def generate(self) -> float | int | str | None:
        value = choice(self.domain).generate()
        if value is None:
            return None
        if self.type.is_int():
            return int(value)
        if self.type.is_float():
            return float(value)
        return value
    
    def get_validator_expr(self, mnemonic: str) -> Column:
        if len(self.domain) == 0:
            warnings.warn(f"Feature {mnemonic + ':' + self.name} has no registered domain. All values will be valid.")
            return F.lit(True) 
        expressions = [domain.get_validator_expr(mnemonic + "_" + self.name) for domain in self.domain]
        return reduce(lambda a, b: a | b, expressions)


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

    def generate(
        self, eps: float = 1e-3, inf_proxy: float = 1e5, decimals: int = 2
    ) -> float:
        start = self.start if self.start != float("-inf") else -inf_proxy
        end = self.end if self.end != float("inf") else inf_proxy
        if self.include_start:
            start += eps
        if self.include_end:
            end -= eps
        return round(uniform(start, end), decimals)

    def get_validator_expr(self, feature_name: str) -> Column:
        conditions = []

        if self.start != float("-inf"):
            if self.include_start:
                conditions.append(F.col(feature_name) >= self.start)
            else:
                conditions.append(F.col(feature_name) > self.start)
        
        if self.end != float("inf"):
            if self.include_end:
                conditions.append(F.col(feature_name) <= self.end)
            else:
                conditions.append(F.col(feature_name) < self.end)

        return reduce(lambda a, b: a & b, conditions)

class ValueDomain(BaseModel):
    """This class represents the metadata for a value domain of a feature"""

    type: Literal["VALUE"] = "VALUE"
    value: float | int | str
    description: str

    def generate(self) -> float | int | str:
        return self.value

    def get_validator_expr(self, feature_name: str) -> Column:
        return F.col(feature_name) == F.lit(self.value)

class NullDomain(BaseModel):
    """This class represents the metadata for a null domain of a feature"""

    type: Literal["MISSING"] = "MISSING"
    description: str

    def generate(self) -> None:
        return None

    def get_validator_expr(self, feature_name: str) -> Column:
        return F.col(feature_name).isNull()


Domain = Annotated[RangeDomain | ValueDomain | NullDomain, Field(discriminator="type")]
