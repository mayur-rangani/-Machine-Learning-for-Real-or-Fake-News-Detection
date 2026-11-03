
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

df["clean_text"] = df["text"].str.lower()
df["clean_text"] = df["clean_text"].str.replace(
    r"[^a-z\s]", " ", regex=True
)
df["clean_text"] = df["clean_text"].str.replace(
    r"\s+", " ", regex=True
)

df = df[df["clean_text"].str.strip() != ""]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df["clean_text"])

nmf = NMF(
    n_components=5,
    random_state=42,
    max_iter=500
)

W = nmf.fit_transform(X)

words = vectorizer.get_feature_names_out()

topics = []

print("\nNMF TOPICS")

for i, topic in enumerate(nmf.components_):

    top_words = topic.argsort()[-10:][::-1]

    topic_words = words[top_words].tolist()

    topics.append(topic_words)

    print("\nTopic", i + 1, ":", topic_words)


df["topic_strength"] = W.max(axis=1) / (W.sum(axis=1) + 1e-10)

print("\nTOPIC RELEVANCE COMPARISON")

print(df.groupby("label")["topic_strength"].mean())


tokenized_text = df["clean_text"].str.split().tolist()

dictionary = Dictionary(tokenized_text)

real_text = df[df["label"] == "real"]["clean_text"].str.split().tolist()

fake_text = df[df["label"] == "fake"]["clean_text"].str.split().tolist()

real_coherence = CoherenceModel(
    topics=topics,
    texts=real_text,
    dictionary=dictionary,
    coherence="c_v",
    processes=1
)

fake_coherence = CoherenceModel(
    topics=topics,
    texts=fake_text,
    dictionary=dictionary,
    coherence="c_v",
    processes=1
)

print("\nTOPIC COHERENCE COMPARISON")

print("Real News:", real_coherence.get_coherence())

print("Fake News:", fake_coherence.get_coherence())
