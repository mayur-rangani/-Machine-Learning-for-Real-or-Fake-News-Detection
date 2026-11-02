import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("dataset/fake_news_dataset.csv")

def count_voice(text):

    doc = nlp(str(text))

    active = 0
    passive = 0

    for token in doc:

        if token.dep_ == "nsubjpass":
            passive = passive + 1

        elif token.dep_ == "nsubj":
            active = active + 1

    return pd.Series([active, passive])


df[["Active_Count", "Passive_Count"]] = df["text"].apply(count_voice)

result = df.groupby("label")[["Active_Count", "Passive_Count"]].mean()

print(result)