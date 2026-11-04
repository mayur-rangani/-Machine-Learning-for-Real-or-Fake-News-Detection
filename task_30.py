
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from rake_nltk import Rake
from collections import Counter

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

df = pd.read_csv("dataset/fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])

df["label"] = df["label"].str.lower().str.strip()

rake = Rake()

real_keywords = Counter()
fake_keywords = Counter()

for i, row in df.iterrows():

    rake.extract_keywords_from_text(row["text"])
    keywords = rake.get_ranked_phrases()[:5]

    if row["label"] == "real":
        real_keywords.update(keywords)

    elif row["label"] == "fake":
        fake_keywords.update(keywords)


comparison = pd.DataFrame({
    "Real News": real_keywords,
    "Fake News": fake_keywords
}).fillna(0)

comparison["Total"] = comparison["Real News"] + comparison["Fake News"]

comparison = comparison.sort_values("Total", ascending=False).head(15)

print("\nTOP KEYWORDS COMPARISON")
print(comparison.drop(columns="Total"))

comparison[["Real News", "Fake News"]].plot(
    kind="barh",
    figsize=(10, 7)
)


