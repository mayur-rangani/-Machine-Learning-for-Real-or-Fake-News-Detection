
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("")
df["label"] = df["label"].fillna("Unknown")

word_count = []
sentence_count = []
avg_word_length = []
vocabulary_diversity = []
sentiment = []

analyzer = SentimentIntensityAnalyzer()

for text in df["text"]:

    words = re.findall(r"\b\w+\b", text.lower())

    word_count.append(len(words))

    sentences = re.split(r'[.!?]+', text)
    count = 0

    for sentence in sentences:
        if sentence.strip() != "":
            count += 1

    sentence_count.append(count)

    total_length = 0

    for word in words:
        total_length += len(word)

    if len(words) > 0:
        avg_word_length.append(total_length / len(words))
    else:
        avg_word_length.append(0)

    if len(words) > 0:
        vocabulary_diversity.append(len(set(words)) / len(words))
    else:
        vocabulary_diversity.append(0)

    score = analyzer.polarity_scores(text)
    sentiment.append(score["compound"])

df["Word_Count"] = word_count
df["Sentence_Count"] = sentence_count
df["Average_Word_Length"] = avg_word_length
df["Vocabulary_Diversity"] = vocabulary_diversity
df["Sentiment"] = sentiment

result = df.groupby("label")[
    ["Word_Count", "Sentence_Count",
     "Average_Word_Length", "Vocabulary_Diversity",
     "Sentiment"]
].mean()

print("\nLinguistic Profiling:")
print(result)

vectorizer = TfidfVectorizer(max_features=1000)

tfidf_matrix = vectorizer.fit_transform(df["text"])

similarity = cosine_similarity(tfidf_matrix)

print("\nCosine Similarity Matrix:")
print(similarity)


