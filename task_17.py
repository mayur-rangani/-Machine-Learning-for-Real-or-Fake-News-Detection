import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")
df["label"] = df["label"].str.lower().str.strip()

real_news = df[df["label"] == "real"]["text"]
fake_news = df[df["label"] == "fake"]["text"]

for n in [2, 3]:

    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words="english"
    )

    real_data = vectorizer.fit_transform(real_news)
    phrases = vectorizer.get_feature_names_out()

    fake_data = vectorizer.transform(fake_news)

    result = pd.DataFrame({
        "Phrase": phrases,
        "Real_Count": real_data.sum(axis=0).A1,
        "Fake_Count": fake_data.sum(axis=0).A1
    })

    result["Total_Count"] = (
        result["Real_Count"] + result["Fake_Count"]
    )

    result = result.sort_values(
        "Total_Count", ascending=False
    )
print(result.head(10))