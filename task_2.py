
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error

df = pd.read_csv("dataset/fake_news_dataset.csv")

print("Missing Values:")
print(df.isnull().sum())

df[["source", "author"]] = df[["source", "author"]].fillna("Unknown")

data = pd.DataFrame({
    "title_length": df["title"].fillna("").str.len(),
    "text_length": df["text"].fillna("").str.len(),
    "word_count": df["text"].fillna("").str.split().str.len()
}).astype(float)


np.random.seed(42)
test_index = np.random.choice(data.index, int(len(data) * 0.1), replace=False)
original = data.loc[test_index, "text_length"].copy()

test_data = data.copy()
test_data.loc[test_index, "text_length"] = np.nan

methods = {
    "Mean": SimpleImputer(strategy="mean"),
    "Median": SimpleImputer(strategy="median"),
    "KNN": KNNImputer(n_neighbors=5),
    "Iterative": IterativeImputer(random_state=42)
}

results = []

for name, imputer in methods.items():
    imputed = pd.DataFrame(
        imputer.fit_transform(test_data),
        columns=test_data.columns
    )

    rmse = np.sqrt(mean_squared_error(
        original, imputed.loc[test_index, "text_length"]
    ))

    results.append([name, round(rmse, 2), imputed.isnull().sum().sum()])

print("\nImputation Comparison:")
print(pd.DataFrame(results, columns=["Method", "RMSE", "Missing Values"]).sort_values("RMSE"))
