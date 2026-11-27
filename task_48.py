
import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

stop_words = set(stopwords.words("english"))

sentences = []

for text in df["text"]:

    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    sentences.append(words)

print(sentences[:5])
