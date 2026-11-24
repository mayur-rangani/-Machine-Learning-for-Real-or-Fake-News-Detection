
import pandas as pd
import re
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

sensational_words = [
    "shocking",
    "breaking",
    "unbelievable",
    "secret",
    "exposed",
    "explosive",
    "urgent",
    "miracle",
    "scandal",
    "incredible",
    "you won't believe",
    "hidden truth",
    "biggest ever",
    "must see",
    "wake up"
]

def analyze_text(text):
    text = text.lower()
    count = 0

    for word in sensational_words:
        pattern = r"\b" + re.escape(word) + r"\b"
        count = count + len(re.findall(pattern, text))

    return count

df["Sensational_Count"] = df["text"].apply(analyze_text)

result = df.groupby("label")["Sensational_Count"].mean()

print("Average Sensationalist Words:")
print(result)

print("\nArticles with Sensationalist Words:")
print(df[["label", "Sensational_Count"]].head(10))

