
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nrclex import NRCLex

df = pd.read_csv("dataset/fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])

emotions = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]

emotion_data = []

for article in df["text"]:

    emotion = NRCLex()
    emotion.load_raw_text(str(article))

    scores = emotion.raw_emotion_scores

    row = {}

    for category in emotions:
        row[category] = scores.get(category, 0)

    emotion_data.append(row)

emotion_df = pd.DataFrame(emotion_data, index=df.index)
df[emotions] = emotion_df

prevalence = df.groupby("label")[emotions].apply(
    lambda x: (x > 0).mean() * 100
)

print("\nEMOTION PREVALENCE COMPARISON")
print(prevalence.round(2))

prevalence.plot(kind="bar", figsize=(12, 6))

plt.title("Emotion Prevalence: Real vs Fake News")
plt.xlabel("News Category")
plt.ylabel("Articles Containing Emotion (%)")
plt.xticks(rotation=0)
plt.legend(title="Emotion")
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 5))

sns.heatmap(
    prevalence,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd"
)

plt.title("Emotion Comparison Heatmap")
plt.xlabel("Emotion Categories")
plt.ylabel("News Category")
plt.tight_layout()
plt.show()
