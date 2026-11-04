
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_csv("dataset/fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf = vectorizer.fit_transform(df["text"])

nmf = NMF(n_components=5, random_state=42)
topics = nmf.fit_transform(tfidf)

df["topic"] = topics.argmax(axis=1)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(nmf.components_):
    top_words = topic.argsort()[-10:][::-1]
    print("Topic", i + 1, ":", [words[j] for j in top_words])

comparison = pd.crosstab(
    df["topic"], df["label"], normalize="columns"
) * 100

print("\nTopic Distribution (%):")
print(comparison.round(2))
