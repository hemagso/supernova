from pyspark.sql import SparkSession
from supernova.query import query_features


def main():
    spark = SparkSession.builder.appName("FeatureSetQuery").getOrCreate()
    population_df = spark.read.parquet("data/parquet/population_df")
    feature_a_df = spark.read.parquet("data/parquet/feature_a_df")
    feature_b_df = spark.read.parquet("data/parquet/feature_b_df")

    features = query_features(
        population_df,
        {"a": feature_a_df, "b": feature_b_df},
        features={"a": ["feature_a", "feature_c"]},
    )
    features.show()
    features.write.parquet("data/parquet/features", mode="overwrite")
    features.coalesce(1).write.csv("data/csv/features", header=True, mode="overwrite")


if __name__ == "__main__":
    main()
