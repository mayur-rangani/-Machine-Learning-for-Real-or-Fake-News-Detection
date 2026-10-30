import pandas as pd
from textblob import TextBlob

df = pd.read_csv("dataset/fake_news_dataset.csv")

polarity = []
subjectivity = []

for text in df["text"]:

    analysis = TextBlob(str(text))

    polarity.append(analysis.sentiment.polarity)

    subjectivity.append(analysis.sentiment.subjectivity)

df["Polarity"] = polarity
df["Subjectivity"] = subjectivity

result = df.groupby("label")[["Polarity", "Subjectivity"]].mean()

print(result)
