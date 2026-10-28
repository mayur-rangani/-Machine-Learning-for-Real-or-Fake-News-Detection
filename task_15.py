import pandas as pd
import spacy
import matplotlib.pyplot as plt


df = pd.read_csv("dataset/fake_news_dataset.csv")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

real_length = []
fake_length = []
real_subjects = 0
fake_subjects = 0
real_objects = 0
fake_objects = 0

for index, row in df.iterrows():

    text = str(row["title"]) + " " + str(row["text"])
    doc = nlp(text)
    label = str(row["label"]).lower()

    for sentence in doc.sents:

        word_count = len(sentence)

        if label == "real":
            real_length.append(word_count)
        elif label == "fake":
            fake_length.append(word_count)

        for token in sentence:

            if token.dep_ == "nsubj":
                if label == "real":
                    real_subjects += 1
                elif label == "fake":
                    fake_subjects += 1

            if token.dep_ in ["dobj", "obj"]:
                if label == "real":
                    real_objects += 1
                elif label == "fake":
                    fake_objects += 1

result = pd.DataFrame({
    "Measure": [
        "Average Sentence Length",
        "Subject Relationships",
        "Object Relationships"
    ],
    "Real News": [
        sum(real_length) / len(real_length) if real_length else 0,
        real_subjects,
        real_objects
    ],
    "Fake News": [
        sum(fake_length) / len(fake_length) if fake_length else 0,
        fake_subjects,
        fake_objects
    ]
})

print(result.round(2))


plt.bar(
    ["Real News", "Fake News"],
    [
        sum(real_length) / len(real_length) if real_length else 0,
        sum(fake_length) / len(fake_length) if fake_length else 0
    ]
)

plt.title("Average Sentence Length Comparison")
plt.xlabel("News Type")
plt.ylabel("Average Number of Words")
plt.show()