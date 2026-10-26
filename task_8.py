
import pandas as pd
import nltk
import spacy
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")

df = pd.read_csv("dataset/fake_news_dataset.csv")
df["text"] = df["text"].fillna("").str.lower()

stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

df["stemming"] = ""

for i in range(len(df)):
    words = nltk.word_tokenize(df.loc[i, "text"])
    stemmed_words = []

    for word in words:
        stemmed_words.append(stemmer.stem(word))

    df.loc[i, "stemming"] = " ".join(stemmed_words)

df["lemmatization"] = ""

for i in range(len(df)):
    doc = nlp(df.loc[i, "text"])
    lemmatized_words = []

    for token in doc:
        if token.is_alpha:
            lemmatized_words.append(token.lemma_)

    df.loc[i, "lemmatization"] = " ".join(lemmatized_words)

print(df[["text", "stemming", "lemmatization"]].head())

df.to_csv("dataset/stemming_lemmatization.csv", index=False)
