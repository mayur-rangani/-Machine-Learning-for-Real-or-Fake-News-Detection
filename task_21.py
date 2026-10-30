import pandas as pd
import textstat

# Load dataset
df = pd.read_csv("dataset/fake_news_dataset.csv")

flesch_scores = []
fog_scores = []

for text in df["text"]:

    text = str(text)

    flesch = textstat.flesch_kincaid_grade(text)
    
    fog = textstat.gunning_fog(text)

    flesch_scores.append(flesch)
    fog_scores.append(fog)

df["Flesch_Kincaid"] = flesch_scores
df["Gunning_Fog"] = fog_scores

result = df.groupby("label")[[
    "Flesch_Kincaid",
    "Gunning_Fog"
]].mean()

print(result)

