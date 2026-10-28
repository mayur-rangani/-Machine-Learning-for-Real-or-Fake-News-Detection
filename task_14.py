import pandas as pd
import spacy
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/fake_news_dataset.csv")


nlp = spacy.load("en_core_web_sm")

real_nouns = []
fake_nouns = []
real_verbs = []
fake_verbs = []

for index, row in df.iterrows():

    text = str(row["title"]) + " " + str(row["text"])
    doc = nlp(text)

    for chunk in doc.noun_chunks:
        if str(row["label"]).lower() == "real":
            real_nouns.append(chunk.text.lower())
        elif str(row["label"]).lower() == "fake":
            fake_nouns.append(chunk.text.lower())

    for token in doc:
        if token.pos_ == "VERB":
            if str(row["label"]).lower() == "real":
                real_verbs.append(token.lemma_.lower())
            elif str(row["label"]).lower() == "fake":
                fake_verbs.append(token.lemma_.lower())


print("Common Noun Phrases in Real News")
print(pd.Series(real_nouns).value_counts().head(10))

print("\nCommon Noun Phrases in Fake News")
print(pd.Series(fake_nouns).value_counts().head(10))

print("\nCommon Verbs in Real News")
print(pd.Series(real_verbs).value_counts().head(10))

print("\nCommon Verbs in Fake News")
print(pd.Series(fake_verbs).value_counts().head(10))

real_top = pd.Series(real_verbs).value_counts().head(10)
fake_top = pd.Series(fake_verbs).value_counts().head(10)

verb_result = pd.DataFrame({
    "Real News": real_top,
    "Fake News": fake_top
}).fillna(0)

verb_result.plot(kind="bar", figsize=(10, 5))
plt.title("Common Verb Comparison: Real vs Fake News")
plt.xlabel("Verbs")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()