
import pandas as pd
import re
import string

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("").str.lower()

df["text"] = df["text"].str.replace(r'\d+', '', regex=True)

df["text"] = df["text"].str.replace(r'[^\w\s]', '', regex=True)

df["text"] = df["text"].str.replace(r'[^a-zA-Z\s]', '', regex=True)

df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

print(df["text"].head())

