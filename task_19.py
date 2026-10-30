import pandas as pd
import re


df = pd.read_csv("dataset/fake_news_dataset.csv")


results = []

for index, row in df.iterrows():

    text = str(row["text"])

    text = text.lower()

   
    words = re.findall(r'\b[a-z]+\b', text)

    
    total_words = len(words)

    unique_words = len(set(words))

   
    if total_words > 0:
        ttr = unique_words / total_words
    else:
        ttr = 0

    hapax = 0

    for word in set(words):
        if words.count(word) == 1:
            hapax = hapax + 1

    if total_words > 0:
        avg_word_length = sum(len(word) for word in words) / total_words
    else:
        avg_word_length = 0

    results.append({
        "TTR": ttr,
        "Unique_Words": unique_words,
        "Total_Words": total_words,
        "Hapax_Legomena": hapax,
        "Average_Word_Length": avg_word_length
    })


features = pd.DataFrame(results)

df = pd.concat([df, features], axis=1)

print(df.groupby("label")[[
    "TTR",
    "Unique_Words",
    "Hapax_Legomena",
    "Average_Word_Length"
]].mean())

