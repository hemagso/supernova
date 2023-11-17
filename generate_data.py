from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date, lit, udf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from supernova.metadata import FeatureStore, Entity, FeatureSet
from devtools import debug
from uuid import uuid4
from typing import Literal
from functools import reduce

MISSING_ENTITY_RATE = 0.1


def generate_entities(spark: SparkSession, entity: Entity, n: int) -> DataFrame:
    entities = [tuple(uuid4().hex for _ in entity.keys) for _ in range(n)]
    df = spark.createDataFrame(entities, entity.keys)
    return df


def generate_features_date(
    entity_df: DataFrame, feature_set: FeatureSet, dt: date
) -> DataFrame:
    df = entity_df.sample(1 - MISSING_ENTITY_RATE)
    for feature in feature_set.features:
        generator = udf(lambda _: feature.generate(), feature.type.get_spark())
        df = df.withColumn(feature.name, generator(lit(1)))
    df = df.withColumn(feature_set.date_field, to_date(lit(dt)))
    return df


def generate_date_list(
    start: date, end: date, period: Literal["day", "week", "month"], offset: int = 0
) -> list[date]:
    start = start + timedelta(days=offset)
    end = end + timedelta(days=offset)
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        if period == "day":
            current = current + relativedelta(days=1)
        elif period == "week":
            current = current + relativedelta(weeks=1)
        elif period == "month":
            current = current + relativedelta(months=1)
    return dates


def generate_features(
    entity_df: DataFrame,
    feature_set: FeatureSet,
    start: date,
    end: date,
    period: Literal["day", "week", "month"],
) -> DataFrame:
    date_list = generate_date_list(start, end, period)
    dfs = [generate_features_date(entity_df, feature_set, dt) for dt in date_list]
    df = reduce(lambda a, b: a.union(b), dfs)
    return df


def main():
    spark = (
        SparkSession.builder.appName("test_data_generation")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    metadata = FeatureStore.from_folder("./docs/example")

    entities_df = {
        entity.name: generate_entities(spark, entity, 1_000)
        for entity in metadata.entities
    }

    features_df = {
        feature_set: generate_features(
            entities_df[feature_set.entity.name],
            feature_set,
            date(2021, 1, 1),
            date(2023, 12, 31),
            "week" if "weekly" in feature_set.tags else "month",
        )
        for feature_set in metadata.feature_sets
    }
    debug(features_df)
    for feature_set, df in features_df.items():
        df.write.mode("overwrite").partitionBy(feature_set.date_field).parquet(
            feature_set.path
        )
    for entity, df in entities_df.items():
        df.write.mode("overwrite").parquet(f"data/parquet/{entity}")


if __name__ == "__main__":
    main()
