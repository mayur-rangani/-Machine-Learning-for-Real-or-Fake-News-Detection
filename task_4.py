import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df=pd.read_csv("dataset/fake_news_dataset.csv")


df=df.dropna(subset=["text"])

df["word_count"]=df["text"].apply(lambda x: len(str(x).split()))

df["character_count"]=df["text"].apply(lambda x: len(str(x)))

df["stopword_count"]=df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in ENGLISH_STOP_WORDS]))

print("Statistical Analysis")
print(df[["word_count", "character_count", "stopword_count"]].describe())


if "label" in df.columns:
    print(df.groupby("label")[["word_count", "character_count",
                               "stopword_count"]].mean())

    df.boxplot(column="word_count", by="label")
    plt.title("Word Count of Real and Fake News")
    plt.suptitle("")
    plt.show()

words = " ".join(df["text"].astype(str)).lower().split()
stopwords = [word for word in words if word in ENGLISH_STOP_WORDS]

print("\nTop 10 Stopwords")
print(pd.Series(stopwords).value_counts().head(10))