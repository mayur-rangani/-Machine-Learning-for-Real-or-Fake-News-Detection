
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")

source_count = df["source"].fillna("Unknown").value_counts()

print("Most Frequent News Sources:")
print(source_count.head(10))


credibility_data = {
    "Source A": 90,
    "Source B": 75,
    "Source C": 40,
    "Unknown": 0
}

df["Credibility Score"] = (
    df["source"].fillna("Unknown").map(credibility_data).fillna(0)
)

df["Credibility"] = df["Credibility Score"].apply(
    lambda x: "High" if x >= 70 else
    ("Medium" if x >= 40 else "Low")
)

print("\nSource Credibility Analysis:")

result = df.groupby("source").agg(
    Total_Articles=("source", "count"),
    Average_Credibility=("Credibility Score", "mean")
)

print(result.sort_values(
    by="Total_Articles", ascending=False
).head(10))

source_count.head(10).plot(kind="bar")

plt.title("Most Frequent News Sources")
plt.xlabel("News Sources")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


