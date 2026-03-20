from src.kmeans_segmentation import load_data, preprocess, train_kmeans, evaluate

df = load_data("data/customers.csv")
X = preprocess(df)

model, labels = train_kmeans(X, 3)

metrics = evaluate(X, labels)

print("Clustering Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
