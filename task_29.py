
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


df = pd.read_csv("dataset/fake_news_dataset.csv")
df = df.dropna(subset=["text", "label"])

real_text = " ".join(df[df["label"].str.lower() == "real"]["text"])
fake_text = " ".join(df[df["label"].str.lower() == "fake"]["text"])

real_wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=STOPWORDS
).generate(real_text)

fake_wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=STOPWORDS
).generate(fake_text)

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(real_wordcloud, interpolation="bilinear")
plt.title("Real News Word Cloud")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(fake_wordcloud, interpolation="bilinear")
plt.title("Fake News Word Cloud")
plt.axis("off")

plt.tight_layout()
plt.show()
