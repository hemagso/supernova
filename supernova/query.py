from __future__ import annotations
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    datediff,
    row_number,
    min,
    max,
)
from pyspark.sql.window import Window
from datetime import timedelta
from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata import FeatureStore, FeatureSet, Feature
import re


class Query:
    def __init__(self, store: FeatureStore, query: list[str]):
        self.store = store
        self.specs = self.parse(query)

    def parse_set(self, item: str) -> tuple[FeatureSet, str]:
        set_mnemonic, feature_spec = item.split(":")
        feature_set = self.store.get_feature_set(set_mnemonic)
        return feature_set, feature_spec

    def parse_feature(self, feature_set, feature_spec: str) -> list[Feature]:
        if feature_spec == "*":
            return feature_set.features
        pattern = re.compile(feature_spec)
        return list(filter(lambda f: pattern.match(f.name), feature_set.features))

    def parse(self, query: list[str]) -> list[tuple[FeatureSet, list[Feature]]]:
        specs_raw = [self.parse_set(item) for item in query]
        specs_parse = [
            (feature_set, self.parse_feature(feature_set, feature_spec))
            for feature_set, feature_spec in specs_raw
        ]
        ans: dict[FeatureSet, list[Feature]] = {}
        for feature_set, features in specs_parse:
            if feature_set.mnemonic not in ans:
                ans[feature_set] = []
            ans[feature_set] += features
        return list(
            (feature_set, features)
            for feature_set, features in ans.items()
            if len(features) > 0
        )

    def execute(
        self, spark: SparkSession, entity_df: DataFrame, pop_date_field: str
    ) -> DataFrame:
        # Loading all dataframes
        feature_dfs = {
            feature_set.mnemonic: spark.read.parquet(feature_set.path)
            for feature_set, _ in self.specs
        }

        # Getting time range
        min_date, max_date = get_time_window(entity_df, 30, pop_date_field)  # Todo: Variable max lag
        queries = {
            feature_set: query_feature_set(
                entity_df,
                pop_date_field,
                feature_df,
                feature_set.date_field,
                feature_set.entity.keys,
                30,
                min_date,
                max_date,
                [f.name for f in features],
                mnemonic=name,
            )
            for (feature_set, features), (name, feature_df) in zip(
                self.specs, feature_dfs.items()
            )
        }

        result = entity_df
        for feature_set, df in queries.items():
            result = result.join(df, on=feature_set.entity.keys + [pop_date_field])

        return result


def filter_time_window(
    feature_df: DataFrame, min_date: str, max_date: str, set_date_field: str
) -> DataFrame:
    """This function filters a dataframe to a time range

    Args:
        feature_df (DataFrame): The dataframe to filter
        min_date (str): The minimum date to filter
        max_date (str): The maximum date to filter
        set_date_field (str): The name of the date field

    Returns:
        DataFrame: The dataframe filtered to the time range
    """
    return feature_df.filter(
        (col(set_date_field) >= min_date) & (col(set_date_field) <= max_date)
    )


def get_time_window(
    population_df: DataFrame, max_lag: int, pop_date_field: str
) -> tuple[str, str]:
    """This function calculates the time window for a feature set query

    Args:
        population_df (DataFrame): The population dataframe
        max_lag (int): The maximum lag to consider
        pop_date_field (str): The name of the date field

    Returns:
        tuple[str, str]: The minimum and maximum dates for the time window
    """
    range_df = population_df.agg(
        min(pop_date_field).alias("min_date"),
        max(pop_date_field).alias("max_date"),
    ).collect()[0]
    min_date = range_df.min_date - timedelta(days=max_lag)
    max_date = range_df.max_date + timedelta(days=max_lag)
    return min_date, max_date


def add_mnemonic(
    df: DataFrame, mnemonic: str, entity_keys: list[str], pop_date_field: str
) -> DataFrame:
    """This functions add a mnemonic prefix to all fields in a dataframe but the
    entity key and observation date

    Args:
        df (DataFrame): The dataframe to add the mnemonic to
        mnemonic (str): The mnemonic to add
        entity_keys (list[str]): The entity keys
        pop_date_field (str): The name of the date field

    Returns:
        DataFrame: The dataframe with the mnemonic added
    """
    return df.select(
        [
            *[col(k) for k in entity_keys],
            col(pop_date_field),
            *[
                col(field.name).alias(f"{mnemonic}_{field.name}")
                for field in df.schema.fields
                if field.name != "entity_id" and field.name != "observation_date"
            ],
        ]
    )


def query_feature_set(
    population_df: DataFrame,
    pop_date_field: str,
    feature_df: DataFrame,
    set_date_field: str,
    entity_keys: list[str],
    max_lag: int,
    min_date: str,
    max_date: str,
    features: list[str] | None = None,
    mnemonic: str | None = None,
) -> DataFrame:
    """This function queries a single feature set

    Args:
        population_df (DataFrame): The population dataframe
        pop_date_field (str): The name of the date field inthe entity dataframe
        feature_df (DataFrame): The feature dataframe
        set_date_field (str): The name of the date field in the feature set
        entity_keys (list[str]): The entity keys
        max_lag (int): The maximum lag to consider
        min_date (str): The minimum date to filter
        max_date (str): The maximum date to filter
        fields (list[str], optional): The fields to return. Defaults to None (All fields).

    Returns:
        DataFrame: The dataframe containing the features
    """
    feature_df = filter_time_window(feature_df, min_date, max_date, set_date_field)
    if features is not None:
        feature_df = feature_df.select(features + entity_keys + [set_date_field])
    feature_df = feature_df.sort(*entity_keys).repartition(*entity_keys)
    population_df = population_df.sort(*entity_keys).repartition(*entity_keys)

    join_condition = reduce(
        lambda c1, c2: c1 & c2,
        [col(f"p.{key}") == col(f"f.{key}") for key in entity_keys],
    )

    joined_df = (
        population_df.alias("p")
        .join(
            feature_df.alias("f"),
            join_condition
            & (col(f"p.{pop_date_field}") > col(f"f.{set_date_field}"))
            & (
                datediff(col(f"p.{pop_date_field}"), col(f"f.{set_date_field}"))
                <= max_lag
            ),
            how="left",
        )
        .drop(*[col(f"f.{key}") for key in entity_keys])
    )
    window_spec = Window.partitionBy(*entity_keys, pop_date_field).orderBy(
        col(f"f.{set_date_field}").desc()
    )
    ranked_df = joined_df.withColumn("rank", row_number().over(window_spec))

    ranked_df = ranked_df.filter(col("rank") == 1).drop("rank")

    if mnemonic:
        ranked_df = add_mnemonic(ranked_df, mnemonic, entity_keys, pop_date_field)

    return ranked_df


def query_features(
    population_df: DataFrame,
    pop_date_field: str,
    feature_dfs: dict[str, DataFrame],
    set_date_field: str,
    entity_keys: list[str],
    features: dict[str, list[str]],
) -> DataFrame:
    """This function queries a list of feature sets

    Args:
        population_df (DataFrame): The population dataframe
        feature_dfs (dict[str,DataFrame]): The feature dataframes

    Returns:
        DataFrame: The dataframe containing the features
    """
    min_date, max_date = get_time_window(population_df, 30, pop_date_field)
    queries = {
        name: query_feature_set(
            population_df,
            pop_date_field,
            feature_df,
            set_date_field,
            entity_keys,
            30,
            min_date,
            max_date,
            features.get(name),
            mnemonic=name,
        )
        for name, feature_df in feature_dfs.items()
    }

    all_dfs = [population_df] + list(queries.values())

    return reduce(
        lambda df1, df2: df1.join(df2, on=["entity_id", "observation_date"]), all_dfs
    )
