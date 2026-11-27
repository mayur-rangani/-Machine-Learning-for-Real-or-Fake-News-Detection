
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

real_news = df[df["label"] == "real"]["text"]
fake_news = df[df["label"] == "fake"]["text"]


vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20
)

real_matrix = vectorizer.fit_transform(real_news)

real_words = vectorizer.get_feature_names_out()
real_counts = real_matrix.sum(axis=0).A1

real_frequency = pd.DataFrame({
    "Word": real_words,
    "Frequency": real_counts
})

real_frequency = real_frequency.sort_values(
    by="Frequency", ascending=False
)

fake_matrix = vectorizer.fit_transform(fake_news)

fake_words = vectorizer.get_feature_names_out()
fake_counts = fake_matrix.sum(axis=0).A1

fake_frequency = pd.DataFrame({
    "Word": fake_words,
    "Frequency": fake_counts
})

fake_frequency = fake_frequency.sort_values(
    by="Frequency", ascending=False
)

print("Real News Word Frequencies:")
print(real_frequency)

print("\nFake News Word Frequencies:")
print(fake_frequency)

plt.figure(figsize=(10, 5))
plt.bar(real_frequency["Word"], real_frequency["Frequency"])
plt.title("Most Frequent Words in Real News")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(fake_frequency["Word"], fake_frequency["Frequency"])
plt.title("Most Frequent Words in Fake News")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()
