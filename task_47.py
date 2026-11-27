
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report

df = pd.read_csv("dataset/fake_news_dataset.csv")

df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

schemes = {
    "Default TF-IDF": TfidfVectorizer(
        stop_words="english",
        max_features=5000
    ),

    "Binary TF-IDF": TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        binary=True
    ),

    "Sublinear TF-IDF": TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        sublinear_tf=True
    )
}

results = []

for name, vectorizer in schemes.items():

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "Scheme": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

results_df = pd.DataFrame(results)

print(results_df)

best_scheme = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest TF-IDF Scheme:")
print(best_scheme)
