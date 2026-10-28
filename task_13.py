import pandas as pd
import spacy
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/fake_news_dataset.csv")


nlp = spacy.load("en_core_web_sm")

df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")


pos_tags = ["NOUN", "VERB", "ADJ", "ADV"]

real_count = {tag: 0 for tag in pos_tags}
fake_count = {tag: 0 for tag in pos_tags}

for index, row in df.iterrows():
    doc = nlp(row["text"])

    for token in doc:
        if token.pos_ in pos_tags:
            if str(row["label"]).lower() == "real":
                real_count[token.pos_] += 1
            elif str(row["label"]).lower() == "fake":
                fake_count[token.pos_] += 1

result = pd.DataFrame({
    "POS Tag": pos_tags,
    "Real News": [real_count[tag] for tag in pos_tags],
    "Fake News": [fake_count[tag] for tag in pos_tags]
})

print(result)


result.plot(x="POS Tag", y=["Real News", "Fake News"], kind="bar")
plt.title("POS Tag Comparison: Real vs Fake News")
plt.ylabel("Frequency")
plt.xlabel("Parts of Speech")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()