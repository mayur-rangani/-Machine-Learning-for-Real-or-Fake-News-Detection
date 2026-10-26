
import pandas as pd
import re

df = pd.read_csv("dataset/fake_news_dataset.csv")

columns = ["title", "text"]

for col in columns:
    df[col] = df[col].fillna("").astype(str).str.lower()

    df[col] = df[col].str.replace(r"<[^>]*>", " ", regex=True)
    df[col] = df[col].str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    df[col] = df[col].str.replace(r"\d+", " ", regex=True)
    df[col] = df[col].str.replace(r"[^a-z\s']", " ", regex=True)
    df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

print(df[columns].head())

