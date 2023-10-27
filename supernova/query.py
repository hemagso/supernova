from __future__ import annotations
from pyspark.sql import DataFrame
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
        if re.match(r"^\/.*\/$", feature_spec):
            pattern = re.compile(feature_spec)
            return list(filter(lambda f: pattern.match(f.name), feature_set.features))
        return [next(filter(lambda f: f.name == feature_spec, feature_set.features))]

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
        return list(ans.items())


def filter_time_window(
    feature_df: DataFrame, min_date: str, max_date: str
) -> DataFrame:
    """This function filters a dataframe to a time range

    Args:
        feature_df (DataFrame): The dataframe to filter
        min_date (str): The minimum date to filter
        max_date (str): The maximum date to filter

    Returns:
        DataFrame: The dataframe filtered to the time range
    """
    return feature_df.filter(
        (col("calculation_date") >= min_date) & (col("calculation_date") <= max_date)
    )


def get_time_window(population_df: DataFrame, max_lag: int) -> tuple[str, str]:
    """This function calculates the time window for a feature set query

    Args:
        population_df (DataFrame): The population dataframe
        max_lag (int): The maximum lag to consider

    Returns:
        tuple[str, str]: The minimum and maximum dates for the time window
    """
    range_df = population_df.agg(
        min("observation_date").alias("min_date"),
        max("observation_date").alias("max_date"),
    ).collect()[0]
    min_date = range_df.min_date - timedelta(days=max_lag)
    max_date = range_df.max_date + timedelta(days=max_lag)
    return min_date, max_date


def add_mnemonic(df: DataFrame, mnemonic: str) -> DataFrame:
    """This functions add a mnemonic prefix to all fields in a dataframe but the
    entity key and observation date

    Args:
        df (DataFrame): The dataframe to add the mnemonic to
        mnemonic (str): The mnemonic to add

    Returns:
        DataFrame: The dataframe with the mnemonic added
    """
    return df.select(
        [
            col("entity_id"),
            col("observation_date"),
            *[
                col(field.name).alias(f"{mnemonic}_{field.name}")
                for field in df.schema.fields
                if field.name != "entity_id" and field.name != "observation_date"
            ],
        ]
    )


def query_feature_set(
    population_df: DataFrame,
    feature_df: DataFrame,
    max_lag: int,
    min_date: str,
    max_date: str,
    features: list[str] | None = None,
    mnemonic: str | None = None,
) -> DataFrame:
    """This function queries a single feature set

    Args:
        population_df (DataFrame): The population dataframe
        feature_df (DataFrame): The feature dataframe
        max_lag (int): The maximum lag to consider
        min_date (str): The minimum date to filter
        max_date (str): The maximum date to filter
        fields (list[str], optional): The fields to return. Defaults to None (All fields).

    Returns:
        DataFrame: The dataframe containing the features
    """
    feature_df = filter_time_window(feature_df, min_date, max_date)
    if features is not None:
        feature_df = feature_df.select(features + ["entity_id", "calculation_date"])
    feature_df = feature_df.sort("entity_id").repartition("entity_id")
    population_df = population_df.sort("entity_id").repartition("entity_id")

    joined_df = (
        population_df.alias("p")
        .join(
            feature_df.alias("f"),
            (col("p.entity_id") == col("f.entity_id"))
            & (col("p.observation_date") > col("f.calculation_date"))
            & (
                datediff(col("p.observation_date"), col("f.calculation_date"))
                <= max_lag
            ),
            how="left",
        )
        .drop(feature_df.entity_id)
    )

    window_spec = Window.partitionBy("p.entity_id", "p.observation_date").orderBy(
        col("f.calculation_date").desc()
    )
    ranked_df = joined_df.withColumn("rank", row_number().over(window_spec))

    if mnemonic:
        ranked_df = add_mnemonic(ranked_df, mnemonic)

    return ranked_df.filter(col("rank") == 1).drop("rank")


def query_features(
    population_df: DataFrame,
    feature_dfs: dict[str, DataFrame],
    features: dict[str, list[str]],
) -> DataFrame:
    """This function queries a list of feature sets

    Args:
        population_df (DataFrame): The population dataframe
        feature_dfs (dict[str,DataFrame]): The feature dataframes

    Returns:
        DataFrame: The dataframe containing the features
    """
    min_date, max_date = get_time_window(population_df, 30)
    queries = {
        name: query_feature_set(
            population_df,
            feature_df,
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
