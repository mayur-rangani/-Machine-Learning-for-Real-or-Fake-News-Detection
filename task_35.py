
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")

analyzer = SentimentIntensityAnalyzer()

sentiment = []
subjectivity = []
emotional_words = []
exclamation = []
capital_words = []

words_list = ["shocking", "terrible", "dangerous", "amazing",
              "horrible", "fear", "angry", "disaster",
              "wonderful", "unbelievable"]

for text in df["text"]:

    score = analyzer.polarity_scores(text)
    sentiment.append(score["compound"])

    blob = TextBlob(text)
    subjectivity.append(blob.sentiment.subjectivity)

    count = 0
    for word in words_list:
        count = count + text.lower().split().count(word)

    emotional_words.append(count)

    exclamation.append(text.count("!"))

    count = 0
    for word in text.split():
        if word.isupper() and len(word) > 1:
            count = count + 1

    capital_words.append(count)

df["Sentiment"] = sentiment
df["Subjectivity"] = subjectivity
df["Emotional_Words"] = emotional_words
df["Exclamation"] = exclamation
df["Capital_Words"] = capital_words

style = []

for i in range(len(df)):

    if df["Emotional_Words"][i] > 2 or df["Exclamation"][i] > 2:
        style.append("Emotionally Charged")
    else:
        style.append("Neutral")

df["Writing_Style"] = style

result = df.groupby("label")[
    ["Sentiment", "Subjectivity", "Emotional_Words",
     "Exclamation", "Capital_Words"]
].mean()

print("\nWriting Style Analysis:")
print(result)

print("\nWriting Style Count:")
print(df["Writing_Style"].value_counts())

