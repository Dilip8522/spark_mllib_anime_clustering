from clustering.genre_kmeans_clustering import run_kmeans_clustering

print("Choose Option:")
print("1. Run Clustering")
choice = input("Enter 1: ")

if choice == "1":
    run_kmeans_clustering()
else:
    print("Invalid choice.")