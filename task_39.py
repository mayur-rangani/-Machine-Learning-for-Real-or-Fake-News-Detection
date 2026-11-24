
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["title", "text", "label"])

analyzer = SentimentIntensityAnalyzer()

df["Headline_Polarity"] = df["title"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)

df["Article_Polarity"] = df["text"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)

df["Polarity_Shift"] = (
    df["Headline_Polarity"] - df["Article_Polarity"]
)

print("Headline and Article Polarity:")
print(df[[
    "label",
    "Headline_Polarity",
    "Article_Polarity",
    "Polarity_Shift"
]].head(10))

result = df.groupby("label")[
    ["Headline_Polarity", "Article_Polarity"]
].mean()

print("\nAverage Polarity:")
print(result)


print("\nArticles with Large Polarity Shifts:")
print(
    df[abs(df["Polarity_Shift"]) > 0.5][
        ["title", "Headline_Polarity",
         "Article_Polarity", "Polarity_Shift", "label"]
    ].head(10)
)
