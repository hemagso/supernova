from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, date_add, rand, round, col
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("dummy_data_generation").getOrCreate()

# Define base date for demonstration
base_date = "2023-01-01"

# Generate dummy data for population_df
population_df = spark.range(1, 11).withColumn(
    "observation_date", date_add(lit(base_date), col("id").cast(IntegerType()))
)
population_df = population_df.withColumnRenamed("id", "entity_id")

# Generate dummy data for feature_df with some random lags to mimic feature calculation dates
feature_df = spark.range(1, 31, 3).withColumn(
    "calculation_date",
    date_add(
        lit(base_date),
        (col("id") - (rand() * 3).cast(IntegerType())).cast(IntegerType()),
    ),
)
feature_df = feature_df.withColumnRenamed("id", "entity_id")

# Generate random features for feature_df
feature_df = feature_df.withColumn("feature_a", round(rand() * 100, 2))
feature_df = feature_df.withColumn("feature_b", round(rand() * 1000, 2))
feature_df = feature_df.withColumn("feature_c", round(rand() * 10, 2))

# Save to Parquet format (adjust paths as needed)
population_df.write.parquet("data/population_df")
feature_df.write.parquet("data/feature_df")

# Show data for verification
population_df.show()
feature_df.show()
