from supernova.metadata import FeatureStore
from pyspark.sql import SparkSession, Row

fs = FeatureStore.from_folder("./docs/example")

spark = SparkSession.builder.appName("FeatureSetQuery").getOrCreate()

df = spark.createDataFrame([Row(**d) for d in fs.feature_sets[0].generate(10)])
df.show()
