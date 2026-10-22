
import pandas as pd

df = pd.read_csv("dataset/fake_news_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nFirst 5 Records:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nBlank Values:")
print((df.astype("string").apply(lambda x: x.str.strip().eq("").sum())))


print("\nData Types:")
print(df.dtypes)

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nInconsistent Values:")
for col in df.select_dtypes(include="object").columns:
    print(col, df[col].value_counts().head())