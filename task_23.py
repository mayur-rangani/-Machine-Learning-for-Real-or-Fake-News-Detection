
import pandas as pd
import matplotlib.pyplot as plt
import re


df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")

emotions = {
    "Fear": ["fear", "afraid", "scared", "terror", "danger", "threat"],
    "Anger": ["anger", "angry", "hate", "furious", "rage", "attack"],
    "Sadness": ["sad", "unhappy", "cry", "grief", "depressed", "loss"],
    "Joy": ["happy", "joy", "love", "excellent", "wonderful", "success"],
    "Disgust": ["disgust", "awful", "terrible", "horrible", "dirty"],
    "Surprise": ["surprise", "shocking", "amazing", "unexpected", "unbelievable"],
    "Trust": ["trust", "honest", "reliable", "safe", "faith"],
    "Anticipation": ["expect", "hope", "future", "await", "upcoming", "prepare"]
}

def emotion_analysis(text):

    words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

    result = []

    for emotion in emotions:
        count = 0

        for word in emotions[emotion]:
            count = count + words.count(word)

        result.append(count)

    return pd.Series(result)


columns = list(emotions.keys())

df[columns] = df["text"].apply(emotion_analysis)

result = df.groupby("label")[columns].mean()

print("\nEmotional Tone Analysis:")
print(result)

