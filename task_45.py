import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlparse

df = pd.read_csv("dataset/fake_news_dataset.csv")

# Extract URLs
df["URLs"] = df["text"].str.findall(
    r'https?://[^\s]+|www\.[^\s]+'
)

domain_list = []

for urls in df["URLs"]:

    article_domains = []

    for url in urls:

        if url.startswith("www."):
            url = "https://" + url

        domain = urlparse(url).netloc
        domain = domain.replace("www.", "")

        if domain != "":
            article_domains.append(domain)

    domain_list.append(article_domains)

df["Domains"] = domain_list

G = nx.Graph()

for domains in df["Domains"]:

    unique_domains = list(set(domains))

    for i in range(len(unique_domains)):

        for j in range(i + 1, len(unique_domains)):

            source1 = unique_domains[i]
            source2 = unique_domains[j]

            if G.has_edge(source1, source2):
                G[source1][source2]["weight"] += 1

            else:
                G.add_edge(source1, source2, weight=1)

print("Total News Sources:", G.number_of_nodes())
print("Total Connections:", G.number_of_edges())

print("\nMost Connected Sources:")

for domain, connections in sorted(
    G.degree(),
    key=lambda item: item[1],
    reverse=True
)[:10]:

    print(domain, ":", connections)

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1800,
    font_size=8
)

plt.title("News Sharing Network Analysis")
plt.show()