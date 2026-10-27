
import pandas as pd
import spacy
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/fake_news_dataset.csv")

nlp = spacy.load("en_core_web_sm")


print(df.columns)

df["content"] = (
    df["title"].fillna("").astype(str) + " " +
    df["text"].fillna("").astype(str)
)


entities = []

for i, row in df.iterrows():
    doc = nlp(row["content"])

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            entities.append({
                "entity": ent.text,
                "type": ent.label_,
                "news_category": row["label"]
            })

entity_df = pd.DataFrame(entities)


entity_df.to_csv("ner_entities.csv", index=False)


print("\nEntity Distribution:")
print(entity_df.groupby(
    ["news_category", "type"]
).size().unstack(fill_value=0))


entity_counts = entity_df.groupby(
    ["news_category", "type"]
).size().unstack(fill_value=0)

entity_counts.plot(kind="bar", figsize=(9, 5))

plt.title("NER Distribution in Real vs Fake News")
plt.xlabel("News Category")
plt.ylabel("Number of Entities")
plt.xticks(rotation=0)
plt.legend(title="Entity Type")
plt.tight_layout()
plt.show()
