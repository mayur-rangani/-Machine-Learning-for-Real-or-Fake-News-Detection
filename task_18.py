import pandas as pd
import re

# Load dataset
df = pd.read_csv("dataset/fake_news_dataset.csv")

df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(str).str.lower().str.strip()


avg_lengths = []
lexical_richness = []
syntactic_complexity = []

for text in df["text"]:

   
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = []

    for sentence in sentences:
        words = sentence.split()
        sentence_lengths.append(len(words))

   
    if len(sentence_lengths) > 0:
        avg_lengths.append(sum(sentence_lengths) / len(sentence_lengths))
    else:
        avg_lengths.append(0)

    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    if len(words) > 0:
        lexical_richness.append(len(set(words)) / len(words))
    else:
        lexical_richness.append(0)

    connectors = re.findall(
        r'\b(because|although|however|which|while|since|unless|whereas)\b',
        text.lower()
    )

    syntactic_complexity.append(len(connectors))

df["Average_Sentence_Length"] = avg_lengths
df["Lexical_Richness"] = lexical_richness
df["Syntactic_Complexity"] = syntactic_complexity


comparison = df.groupby("label")[
    ["Average_Sentence_Length",
     "Lexical_Richness",
     "Syntactic_Complexity"]
].mean()


print(comparison)