# from pyspark.ml.clustering import KMeans
# from clustering.preprocessing import preprocess_data

# def run_kmeans_clustering():
#     final_df = preprocess_data("anime-dataset-2023.csv")

#     kmeans = KMeans(k=10, seed=42)
#     model = kmeans.fit(final_df.select("features"))
#     predictions = model.transform(final_df)

#     predictions.select("anime_id", "Name", "Genres", "prediction").show(10, truncate=False)

#     # Save clustered output
#     predictions.select("anime_id", "Name", "prediction").write.csv("anime_clusters_output", header=True)

#     # Recommend similar anime example usage:
#     recommend_similar_anime("Death Note", predictions)

# def recommend_similar_anime(anime_name, predictions_df):
#     anime_cluster = predictions_df.filter(predictions_df["Name"] == anime_name).select("prediction").collect()[0][0]
#     similar_anime = predictions_df.filter(predictions_df["prediction"] == anime_cluster)
#     print(f"\nAnime similar to '{anime_name}':")
#     similar_anime.select("Name", "Genres").show(10, truncate=False)


from pyspark.ml.clustering import KMeans
from clustering.preprocessing import preprocess_data

def run_kmeans_clustering():
    final_df = preprocess_data("anime-dataset-2023.csv")

    kmeans = KMeans(k=10, seed=42)
    model = kmeans.fit(final_df.select("features"))
    predictions = model.transform(final_df)

    # Save clustered output
    predictions.select("anime_id", "Name", "Genres", "prediction").write.csv("anime_clusters_output", header=True)

    # Display 10 anime from each cluster
    print("\nShowing 10 anime from each cluster:\n")
    for i in range(10):
        print(f"\nCluster {i}:\n")
        predictions.filter(predictions["prediction"] == i)\
                   .select("Name", "Genres")\
                   .show(10, truncate=False)

    # Optional: Let user choose an anime to get recommendations
    anime_name = input("\nEnter an anime name to get similar recommendations: ")
    recommend_similar_anime(anime_name, predictions)

def recommend_similar_anime(anime_name, predictions_df):
    try:
        anime_cluster = predictions_df.filter(predictions_df["Name"] == anime_name)\
                                      .select("prediction").collect()[0][0]
        similar_anime = predictions_df.filter(predictions_df["prediction"] == anime_cluster)
        print(f"\nAnime similar to '{anime_name}':")
        similar_anime.select("Name", "Genres").show(10, truncate=False)
    except IndexError:
        print(f"\nAnime '{anime_name}' not found in the dataset.")
