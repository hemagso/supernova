from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, lit, round, rand
from pyspark.sql.types import StructType, StructField, StringType, LongType
from datetime import date, timedelta

spark = SparkSession.builder.appName("test_data_generation").getOrCreate()


def generate_population_data():
    # 1. Generate population_df data
    start_date = date(2023, 1, 1)
    end_date = date(2024, 1, 1)
    weeks = [
        (start_date + timedelta(days=i * 7)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days // 7)
    ]

    schema = StructType(
        [
            StructField("observation_date", StringType(), True),
            StructField("dummy", LongType(), True),
        ]
    )

    population_base = spark.createDataFrame(
        [(d, 1) for d in weeks], schema=schema
    ).withColumn("observation_date", to_date("observation_date"))
    entities = spark.range(1, 51).withColumn("dummy", lit(1))  # 50 entities
    population_df = (
        population_base.crossJoin(entities)
        .drop("dummy")
        .withColumnRenamed("id", "entity_id")
    )

    return population_df


def generate_feature_a_data():
    # 2. Generate feature_df data
    start_date_feat = date(2018, 1, 1)
    end_date_feat = date(2024, 1, 1)
    months = [
        (start_date_feat + timedelta(days=i * 30)).strftime("%Y-%m-%d")
        for i in range((end_date_feat - start_date_feat).days // 30)
    ]

    schema = StructType(
        [
            StructField("calculation_date", StringType(), True),
            StructField("dummy", LongType(), True),
        ]
    )

    feature_base = spark.createDataFrame(
        [(d, 1) for d in months], schema=schema
    ).withColumn("calculation_date", to_date("calculation_date"))
    entities = spark.range(1, 501).withColumn("dummy", lit(1))  # 450 entities
    feature_df = (
        feature_base.crossJoin(entities)
        .drop("dummy")
        .withColumnRenamed("id", "entity_id")
    )

    # 3. & 4. Entity mismatch & gaps
    gap_start = date(2023, 5, 1)
    gap_end = date(2023, 9, 1)
    feature_df = feature_df.filter(
        ~(
            (feature_df.entity_id == 5)
            & (feature_df.calculation_date >= gap_start)
            & (feature_df.calculation_date < gap_end)
        )
    )

    # Add random features to feature_df
    feature_df = feature_df.withColumn("feature_a", round(rand() * 100, 2))
    feature_df = feature_df.withColumn("feature_b", round(rand() * 1000, 2))
    feature_df = feature_df.withColumn("feature_c", round(rand() * 10, 2))

    return feature_df


def generate_feature_b_data():
    # 2. Generate feature_df data
    start_date_feat = date(2018, 1, 1)
    end_date_feat = date(2024, 1, 1)
    months = [
        (start_date_feat + timedelta(days=i * 30 + 5)).strftime("%Y-%m-%d")
        for i in range((end_date_feat - start_date_feat).days // 30)
    ]

    schema = StructType(
        [
            StructField("calculation_date", StringType(), True),
            StructField("dummy", LongType(), True),
        ]
    )

    feature_base = spark.createDataFrame(
        [(d, 1) for d in months], schema=schema
    ).withColumn("calculation_date", to_date("calculation_date"))
    entities = spark.range(1, 501).withColumn("dummy", lit(1))  # 450 entities
    feature_df = (
        feature_base.crossJoin(entities)
        .drop("dummy")
        .withColumnRenamed("id", "entity_id")
    )

    # 3. & 4. Entity mismatch & gaps
    gap_start = date(2023, 2, 1)
    gap_end = date(2023, 5, 1)
    feature_df = feature_df.filter(
        ~(
            (feature_df.entity_id == 3)
            & (feature_df.calculation_date >= gap_start)
            & (feature_df.calculation_date < gap_end)
        )
    )

    # Add random features to feature_df
    feature_df = feature_df.withColumn("feature_d", round(rand() * 100, 2))
    feature_df = feature_df.withColumn("feature_e", round(rand() * 1000, 2))
    feature_df = feature_df.withColumn("feature_f", round(rand() * 10, 2))

    return feature_df


# Generate data using the functions
population_df = generate_population_data()
feature_a_df = generate_feature_a_data()
feature_b_df = generate_feature_b_data()

# Show the data
population_df.show()
feature_a_df.show()

population_df.write.parquet("data/parquet/population_df", mode="overwrite")
feature_a_df.write.parquet("data/parquet/feature_a_df", mode="overwrite")
feature_b_df.write.parquet("data/parquet/feature_b_df", mode="overwrite")
