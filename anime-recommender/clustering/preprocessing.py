from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF

def preprocess_data(file_path):
    spark = SparkSession.builder.appName("AnimePreprocessing").getOrCreate()
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Combine 'Genres' and 'Synopsis' columns into a single text field
    df = df.withColumn("text", concat_ws(" ", "Genres", "Synopsis"))
    df = df.withColumn("text", lower(col("text")))

    # Tokenize and remove stopwords
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
    df = tokenizer.transform(df)

    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    df = remover.transform(df)

    # TF-IDF
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    featurized_data = hashingTF.transform(df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized_data)
    final_df = idf_model.transform(featurized_data)

    return final_df