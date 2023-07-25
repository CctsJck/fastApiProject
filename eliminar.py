import pandas as pd

df = pd.read_csv('data.csv', header=0)
print(df)
print(df['angry0'].mean())
