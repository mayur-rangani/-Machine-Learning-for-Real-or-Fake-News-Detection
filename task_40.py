
import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/fake_news_dataset.csv")

df["Statistics"] = df["text"].astype(str).apply(
    lambda x: re.findall(r'\d+(?:\.\d+)?%?', x)
)

df["Statistics Count"] = df["Statistics"].apply(len)

df["Citations"] = df["text"].astype(str).apply(
    lambda x: re.findall(r'https?://\S+|www\.\S+', x)
)

df["Citation Count"] = df["Citations"].apply(len)

df["Statistics Status"] = df["Statistics Count"].apply(
    lambda x: "Check Statistics" if x > 5 else "Normal"
)

df["Citation Status"] = df["Citation Count"].apply(
    lambda x: "Missing Citation" if x == 0 else "Citation Found"
)

print(df[[
    "label",
    "Statistics",
    "Statistics Count",
    "Statistics Status",
    "Citations",
    "Citation Status"
]].head(10))

print("\nReal and Fake News Analysis:")

print(df.groupby("label")[[
    "Statistics Count",
    "Citation Count"
]].mean())


