
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

df = pd.read_csv("dataset/fake_news_dataset.csv")


print("Class Distribution")
print(df["label"].value_counts())


print("\nClass Percentages")
print(df["label"].value_counts(normalize=True) * 100)


counts = df["label"].value_counts()
imbalance_ratio = counts.max() / counts.min()
print("\nImbalance Ratio:", imbalance_ratio)

df["label"].value_counts().plot(kind="bar")
plt.title("Real and Fake News Distribution")
plt.xlabel("News Label")
plt.ylabel("Number of Articles")
plt.show()


majority = df[df["label"] == counts.idxmax()]
minority = df[df["label"] == counts.idxmin()]

majority_down = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

balanced_data = pd.concat([majority_down, minority])
print("\nAfter Undersampling")
print(balanced_data["label"].value_counts())
