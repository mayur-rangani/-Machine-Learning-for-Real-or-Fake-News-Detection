import pandas as pd
import re
from urllib.parse import urlparse

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["URLs"] = df["text"].str.findall(
    r'https?://[^\s]+|www\.[^\s]+'
)

def get_domain(url):
    if url.startswith("www."):
        url = "https://" + url

    domain = urlparse(url).netloc
    return domain.replace("www.", "")

df["Domains"] = df["URLs"].apply(
    lambda urls: [get_domain(url) for url in urls]
)

df["URL_Count"] = df["URLs"].apply(len)

print(df[["label", "URLs", "Domains", "URL_Count"]].head())

all_domains = df["Domains"].explode()

print("\nMost Frequently Used Domains:")
print(all_domains.value_counts().head(10))

print("\nAverage URL Count:")
print(df.groupby("label")["URL_Count"].mean())