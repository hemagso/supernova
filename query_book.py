from pyspark.sql import SparkSession
from supernova.metadata import FeatureStore


def main():
    spark = SparkSession.builder.appName("FeatureSetQuery").getOrCreate()
    store = FeatureStore.from_folder("./docs/example")

    query = store.query(["bci:feature_[a-c]", "bcc:feature_d"])
    features = query.execute(spark, "data/parquet/population_df", "observation_date")
    features = features.sort("entity_id", "observation_date")
    features.show()


if __name__ == "__main__":
    main()
