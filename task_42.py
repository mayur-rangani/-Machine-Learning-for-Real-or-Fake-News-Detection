
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["author"] = df["author"].fillna("Unknown")

author_count = df["author"].value_counts()

print("Most Common Authors:")
print(author_count.head(10))

author_analysis = pd.crosstab(
    df["author"],
    df["label"]
)

print("\nAuthor Analysis:")
print(author_analysis.head(10))

author_analysis["Total Articles"] = author_analysis.sum(axis=1)

if "fake" in author_analysis.columns:
    author_analysis["Fake News Count"] = author_analysis["fake"]
else:
    author_analysis["Fake News Count"] = 0

author_analysis["Fake News Percentage"] = (
    author_analysis["Fake News Count"] /
    author_analysis["Total Articles"] * 100
)

author_analysis["Pattern"] = author_analysis[
    "Fake News Percentage"
].apply(
    lambda x: "High Fake News Pattern" if x >= 70
    else "Normal Pattern"
)

print("\nPotential Authorship Patterns:")

print(
    author_analysis.sort_values(
        by="Fake News Count",
        ascending=False
    ).head(10)
)

