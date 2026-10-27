
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("dataset/fake_news_dataset.csv")
df["text"] = df["text"].fillna("").str.lower()

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []


word_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000))
])

word_model.fit(X_train, y_train)
word_pred = word_model.predict(X_test)

results.append({
    "Tokenization": "Word-level",
    "Accuracy": accuracy_score(y_test, word_pred),
    "F1 Score": f1_score(y_test, word_pred, average="weighted")
})


char_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
    ("model", LogisticRegression(max_iter=1000))
])

char_model.fit(X_train, y_train)
char_pred = char_model.predict(X_test)

results.append({
    "Tokenization": "Character-level",
    "Accuracy": accuracy_score(y_test, char_pred),
    "F1 Score": f1_score(y_test, char_pred, average="weighted")
})

print(pd.DataFrame(results).round(4))
