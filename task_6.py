import pandas as pd

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].str.lower()
df["title"] = df["title"].str.lower()

df["text"] = df["text"].apply(lambda x: " ".join(x.split()))
df["title"] = df["title"].apply(lambda x: " ".join(x.split()))

print(df[["text", "title"]].head())

df.to_csv("dataset/standardized_news_dataset.csv", index=False)

print("Text standardization completed.")