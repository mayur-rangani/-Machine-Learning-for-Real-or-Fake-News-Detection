
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")
df["label"] = df["label"].fillna("Unknown")

analyzer = SentimentIntensityAnalyzer()

positive = []
negative = []
neutral = []
compound = []

fear = []
anger = []
joy = []
sadness = []

fear_words = ["fear", "scared", "danger", "threat",
              "terror", "frightening", "panic"]

anger_words = ["anger", "angry", "hate", "furious",
               "rage", "disgusting", "violence"]

joy_words = ["happy", "joy", "wonderful", "amazing",
             "excellent", "success", "great"]

sadness_words = ["sad", "tragedy", "cry", "death",
                 "pain", "grief", "heartbreaking"]

for text in df["text"]:

    score = analyzer.polarity_scores(text)

    positive.append(score["pos"])
    negative.append(score["neg"])
    neutral.append(score["neu"])
    compound.append(score["compound"])

    words = re.findall(r"\b\w+\b", text.lower())

    fear_count = 0
    anger_count = 0
    joy_count = 0
    sadness_count = 0

    for word in words:

        if word in fear_words:
            fear_count += 1

        if word in anger_words:
            anger_count += 1

        if word in joy_words:
            joy_count += 1

        if word in sadness_words:
            sadness_count += 1

    total = len(words)

    if total > 0:
        fear.append(fear_count / total)
        anger.append(anger_count / total)
        joy.append(joy_count / total)
        sadness.append(sadness_count / total)
    else:
        fear.append(0)
        anger.append(0)
        joy.append(0)
        sadness.append(0)

df["Positive"] = positive
df["Negative"] = negative
df["Neutral"] = neutral
df["Compound"] = compound

df["Fear"] = fear
df["Anger"] = anger
df["Joy"] = joy
df["Sadness"] = sadness

result = df.groupby("label")[
    ["Positive", "Negative", "Neutral", "Compound",
     "Fear", "Anger", "Joy", "Sadness"]
].mean()

print("\nSentiment and Emotion Analysis:")
print(result)
