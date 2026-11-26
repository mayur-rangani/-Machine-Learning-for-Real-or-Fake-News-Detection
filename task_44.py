import pandas as pd
import re
from urllib.parse import urlparse

df = pd.read_csv("dataset/fake_news_dataset.csv")

df["URLs"] = df["text"].str.findall(
    r'https?://[^\s]+|www\.[^\s]+'
)

domains = []

for urls in df["URLs"]:
    for url in urls:

        if url.startswith("www."):
            url = "https://" + url

        domain = urlparse(url).netloc
        domain = domain.replace("www.", "")

        domains.append(domain)

domain_df = pd.DataFrame(domains, columns=["Domain"])

source_count = domain_df["Domain"].value_counts()

print("\nFrequently Used News Sources:")
print(source_count.head(10))

domain_df = df[["label", "URLs"]].copy()

domain_df["Domain"] = df["URLs"].apply(
    lambda urls: [
        urlparse(url if url.startswith("http") else "https://" + url).netloc.replace("www.", "")
        for url in urls
    ]
)

domain_df = domain_df.explode("Domain")
domain_df = domain_df.dropna(subset=["Domain"])

result = domain_df.groupby("Domain")["label"].value_counts().unstack(fill_value=0)

print("\nDomain Credibility Analysis:")
print(result)

trusted_sources = ["bbc.com", "reuters.com", "apnews.com"]

for domain in result.index:

    if domain in trusted_sources:
        print(domain, "- Recognized News Source")

    else:
        print(domain, "- Requires External Verification")