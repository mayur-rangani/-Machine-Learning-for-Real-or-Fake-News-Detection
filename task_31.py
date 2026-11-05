import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

sia = SentimentIntensityAnalyzer()

df["compound"] = df["text"].apply(
    lambda x: sia.polarity_scores(str(x))["compound"]
)

df["positive"] = df["text"].apply(
    lambda x: sia.polarity_scores(str(x))["pos"]
)

df["negative"] = df["text"].apply(
    lambda x: sia.polarity_scores(str(x))["neg"]
)

df["neutral"] = df["text"].apply(
    lambda x: sia.polarity_scores(str(x))["neu"]
)

df["sentiment"] = df["compound"].apply(
    lambda x: "Positive" if x >= 0.05
    else "Negative" if x <= -0.05
    else "Neutral"
)

print("SENTIMENT COMPARISON")
print(df.groupby("label")[["positive", "negative",
                           "neutral", "compound"]].mean())
