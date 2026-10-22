
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset/fake_news_dataset.csv")

print("Original Dataset:", df.shape)


text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower().str.strip()


df = df.loc[~text.duplicated()].reset_index(drop=True)
text = text.loc[df.index] if False else (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower().str.strip()


tfidf = TfidfVectorizer(stop_words="english").fit_transform(text)
similarity = cosine_similarity(tfidf)

remove = set()
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if similarity[i, j] >= 0.90:
            remove.add(j)

clean_df = df.drop(df.index[list(remove)])

print("Cleaned Dataset:", clean_df.shape)
print("Duplicates Removed:", len(pd.read_csv("dataset/fake_news_dataset.csv")) - len(clean_df))

clean_df.to_csv("dataset/cleaned_news_dataset.csv", index=False)
print("Cleaned dataset saved!")
