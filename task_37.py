
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

fake_news = df[df["label"] == "fake"]["text"]
real_news = df[df["label"] == "real"]["text"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20
)

fake_tfidf = vectorizer.fit_transform(fake_news)

words = vectorizer.get_feature_names_out()
scores = fake_tfidf.mean(axis=0).A1

tfidf_result = pd.DataFrame({
    "Word": words,
    "TF-IDF Score": scores
})

tfidf_result = tfidf_result.sort_values(
    by="TF-IDF Score",
    ascending=False
)

print("Important Fake News Words:")
print(tfidf_result)

def word_frequency(data):
    all_words = []

    for article in data:
        article = article.lower()
        article = re.sub(r"[^a-z\s]", "", article)
        all_words.extend(article.split())

    return Counter(all_words)

fake_count = word_frequency(fake_news)
real_count = word_frequency(real_news)

print("\nTop Fake News Words:")
print(fake_count.most_common(10))

print("\nTop Real News Words:")
print(real_count.most_common(10))

