from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, lit, round, rand
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import date, timedelta
from supernova.metadata import FeatureStore, Entity
from devtools import debug
from uuid import uuid4


def generate_entities(spark: SparkSession, entity: Entity, n: int):
    entities = [tuple(uuid4().hex for _ in entity.keys) for _ in range(n)]
    df = spark.createDataFrame(entities, entity.keys)
    return df


def main():
    spark = SparkSession.builder.appName("test_data_generation").getOrCreate()
    metadata = FeatureStore.from_folder("./docs/example")

    entities_df = {
        entity.name: generate_entities(spark, entity, 1_000)
        for entity in metadata.entities
    }
    debug(entities_df)


if __name__ == "__main__":
    main()
