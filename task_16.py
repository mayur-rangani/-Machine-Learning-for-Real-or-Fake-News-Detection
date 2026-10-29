import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")


vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(df["text"])


words = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=words
)

print("TF-IDF values:")
print(tfidf_df.head())


word_scores = tfidf_df.mean().sort_values(ascending=False)

print("\nTop 10 important words:")
print(word_scores.head(10))

