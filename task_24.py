
import pandas as pd
import re
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")


df["text"] = df["text"].fillna("")


hyperbolic_words = [
    "shocking", "unbelievable", "incredible",
    "massive", "explosive", "terrifying",
    "horrifying", "miracle", "disaster",
    "outrageous", "devastating"
]

sensational_phrases = [
    "you won't believe",
    "what happens next",
    "shocking truth",
    "secret revealed",
    "the truth will shock you",
    "everyone is talking about"
]


extreme_words = [
    "always", "never", "everyone",
    "nobody", "completely", "absolutely",
    "all", "nothing"
]


def analyze_sensationalism(text):

    text = str(text)
    lower_text = text.lower()

   
    hyperbolic_count = 0

    for word in hyperbolic_words:
        hyperbolic_count += len(
            re.findall(r"\b" + re.escape(word) + r"\b", lower_text)
        )

    phrase_count = 0

    for phrase in sensational_phrases:
        phrase_count += lower_text.count(phrase)

    extreme_count = 0

    for word in extreme_words:
        extreme_count += len(
            re.findall(r"\b" + re.escape(word) + r"\b", lower_text)
        )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    capital_count = len(re.findall(r"\b[A-Z]{3,}\b", text))

    total_words = len(text.split())

    score = (
        hyperbolic_count * 2
        + phrase_count * 3
        + extreme_count
        + exclamation_count
        + question_count
        + capital_count * 2
    )

    if total_words > 0:
        score = (score / total_words) * 100

    return pd.Series([
        hyperbolic_count,
        phrase_count,
        extreme_count,
        exclamation_count,
        question_count,
        capital_count,
        score
    ])


df[[
    "Hyperbolic_Count",
    "Sensational_Phrase_Count",
    "Extreme_Word_Count",
    "Exclamation_Count",
    "Question_Count",
    "Capital_Count",
    "Sensationalism_Score"
]] = df["text"].apply(analyze_sensationalism)

result = df.groupby("label")[[
    "Hyperbolic_Count",
    "Sensational_Phrase_Count",
    "Extreme_Word_Count",
    "Exclamation_Count",
    "Question_Count",
    "Capital_Count",
    "Sensationalism_Score"
]].mean()

print("\nSensationalism Analysis:")
print(result)


