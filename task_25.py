import pandas as pd
import re


df = pd.read_csv("dataset/fake_news_dataset.csv")


df["exclamation_count"] = df["text"].str.count("!")
df["question_count"] = df["text"].str.count(r"\?")
df["dot_count"] = df["text"].str.count(r"\.")
df["multiple_punctuation"] = df["text"].str.count(r"[!?]{2,}")


df["uppercase_count"] = df["text"].str.count(r"[A-Z]")
df["total_letters"] = df["text"].str.count(r"[A-Za-z]")

df["uppercase_percentage"] = (
    df["uppercase_count"] / df["total_letters"].replace(0, 1)
) * 100


words = ["shocking", "breaking", "urgent",
         "unbelievable", "amazing", "secret"]

df["sensational_count"] = df["text"].str.lower().apply(
    lambda x: sum(x.count(word) for word in words)
)


print(df[[
    "exclamation_count",
    "question_count",
    "multiple_punctuation",
    "uppercase_percentage",
    "sensational_count"
]].head())