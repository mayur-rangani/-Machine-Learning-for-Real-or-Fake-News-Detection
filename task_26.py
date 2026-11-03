import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("dataset/fake_news_dataset.csv")


df = df.dropna(subset=["text", "label"])

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000
)

data = vectorizer.fit_transform(df["text"])

lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)

lda.fit(data)

words = vectorizer.get_feature_names_out()

print("LDA TOPICS - REAL AND FAKE NEWS")

for i, topic in enumerate(lda.components_):

    top_words = topic.argsort()[-10:][::-1]

    print("\nTopic", i + 1, ":")

    for word in top_words:
        print(words[word], end=", ")

    print()