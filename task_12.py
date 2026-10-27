
import pandas as pd
import spacy
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")
nlp = spacy.load("en_core_web_sm")

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

df["entities"] = df["content"].apply(
    lambda x: pd.Series([ent.label_ for ent in nlp(x).ents
                         if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]]).value_counts().to_dict()
)

frequency = pd.DataFrame(df["entities"].tolist()).fillna(0)
frequency["label"] = df["label"].values

result = frequency.groupby("label").sum()

print("Entity Frequency in Real and Fake News:")
print(result)

result.plot(kind="bar", figsize=(8, 5))
plt.title("Entity Frequency: Real vs Fake News")
plt.xlabel("News Category")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
