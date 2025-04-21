# spark_mllib_anime_clustering
This project implements an Anime Recommendation System using the **KMeans Clustering** algorithm in **Apache Spark (PySpark)**. It processes anime metadata (genres and synopsis) using NLP techniques and groups similar anime into clusters, allowing recommendations based on shared clusters.

---

## 🚀 Features

- Anime content preprocessing using NLP (TF-IDF)
- KMeans clustering of anime based on genres and synopsis
- Command-line based user interaction to:
  - Run clustering
  - Enter an anime name to get 10 similar recommendations
- Saves cluster results for visualization and future analysis

---

## 📁 Project Structure
anime-recommender/
│
├── main.py
├── clustering/
│   ├── __init__.py
│   ├── genre_kmeans_clustering.py
│   └── preprocessing.py
│
├── anime-dataset-2023.csv

---

## 🧠 How It Works

1. **Preprocessing (`clustering/preprocessing.py`)**
   - Merges `Genres` and `Synopsis` fields
   - Lowercases text, tokenizes, removes stopwords
   - Applies TF-IDF to convert text to numeric features

2. **Clustering (`clustering/genre_kmeans_clustering.py`)**
   - Uses KMeans to group anime into 10 clusters
   - Saves predictions and displays sample clustered data

3. **Recommendation**
   - Asks the user for an anime name
   - Finds the cluster of that anime
   - Displays 10 other anime from the same cluster

---

## 🖥️ How to Run

> Ensure you have **Apache Spark** and **PySpark** installed.

### Step 1: Clone the repository
in bash
git clone https://github.com/yourusername/anime-recommender.git
cd anime-recommender

### Step 2: Run the Project
python main.py

Follow the prompts:
Choose Option:
1. Run Clustering
Enter 1:
Enter anime name to get recommendations: Death Note

## 📦 Output
Console shows 10 anime in each cluster

Recommendations based on selected anime

Clusters saved in anime_clusters_output/ as CSV

## 📊 Sample Output
Cluster 2 contains:
+--------------------+-------------------------------+
|Name                |Genres                         |
+--------------------+-------------------------------+
|Death Note          |Mystery, Psychological, Thriller|
|Monster             |Thriller, Drama, Psychological |
|Code Geass          |Action, Sci-Fi, Mecha          |
|...                 |...                            |

## 📌 Dependencies
Python 3.x
PySpark

## 📈 Future Improvements
Add fuzzy matching for anime name input

Web interface for easier interaction

Integration of collaborative filtering

Visualization of clusters (e.g., using PCA or t-SNE)

## 📜 License
This project is open-source and free to use under the MIT License.

## 🙌 Acknowledgements
Apache Spark

Anime dataset used: anime-dataset-2023.csv

## 👨‍💻 Author
Dilipkumar
