
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

nltk.download("vader_lexicon")

df = pd.read_csv("dataset/fake_news_dataset.csv")
df = df.dropna(subset=["title", "text", "label"])

sia = SentimentIntensityAnalyzer()

headline_scores = []
article_scores = []

for title in df["title"]:
    score = sia.polarity_scores(str(title))
    headline_scores.append(score["compound"])

for article in df["text"]:
    score = sia.polarity_scores(str(article))
    article_scores.append(score["compound"])

df["Headline"] = headline_scores
df["Article"] = article_scores

print("\nSENTIMENT COMPARISON")
print(df.groupby("label")[["Headline", "Article"]].mean())

df.groupby("label")[["Headline", "Article"]].mean().plot(
    kind="bar", figsize=(9, 5)
)

plt.title("Headline vs Article Sentiment")
plt.xlabel("News Category")
plt.ylabel("Compound Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

emotions = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]

headline_emotions = []
article_emotions = []

for title in df["title"]:

    emotion = NRCLex()
    emotion.load_raw_text(str(title))

    scores = emotion.raw_emotion_scores

    row = {}

    for category in emotions:
        row[category] = scores.get(category, 0)

    headline_emotions.append(row)

for article in df["text"]:

    emotion = NRCLex()
    emotion.load_raw_text(str(article))

    scores = emotion.raw_emotion_scores

    row = {}

    for category in emotions:
        row[category] = scores.get(category, 0)

    article_emotions.append(row)

headline_df = pd.DataFrame(headline_emotions, index=df.index)
article_df = pd.DataFrame(article_emotions, index=df.index)

for category in emotions:
    df["Headline_" + category] = headline_df[category]
    df["Article_" + category] = article_df[category]

results = []

for label in df["label"].unique():

    for emotion in emotions:

        headline_prevalence = (
            df.loc[df["label"] == label, "Headline_" + emotion] > 0
        ).mean() * 100

        article_prevalence = (
            df.loc[df["label"] == label, "Article_" + emotion] > 0
        ).mean() * 100

        results.append({
            "label": label,
            "emotion": emotion,
            "Headline": headline_prevalence,
            "Article": article_prevalence
        })

emotion_df = pd.DataFrame(results)

print("\nEMOTION PREVALENCE")
print(emotion_df.round(2))

plot_df = emotion_df.melt(
    id_vars=["label", "emotion"],
    var_name="Content Type",
    value_name="Prevalence"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="emotion",
    y="Prevalence",
    hue="Content Type"
)

plt.title("Headline vs Article Emotion Prevalence")
plt.xlabel("Emotion")
plt.ylabel("Prevalence (%)")
plt.tight_layout()
plt.show()

df["Difference"] = df["Headline"] - df["Article"]

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="label", y="Difference")

plt.axhline(0, linestyle="--")
plt.title("Headline vs Article Sentiment Difference")
plt.xlabel("News Category")
plt.ylabel("Sentiment Difference")
plt.tight_layout()
plt.show()
