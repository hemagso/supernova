from supernova.metadata import FeatureStore
from supernova.query import query_features
from devtools import debug
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("FeatureSetQuery").getOrCreate()
    path = "./docs/example"
    store = FeatureStore.from_folder(path)
    query = store.query(["bci:*", "bcc:*"])
    feature_sets = {
        feature_set.mnemonic: spark.read.parquet(feature_set.path)
        for feature_set, _ in query.specs
    }
    features = {fs.mnemonic: [f.name for f in features] for fs, features in query.specs}
    debug(features)
    population_df = spark.read.parquet("data/parquet/population_df")
    df = query_features(population_df, feature_sets, features)
    df.show()


if __name__ == "__main__":
    main()
