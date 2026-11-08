import pandas as pd
df = pd.read_csv('data/zomato.csv.zip')
print(df.info())
print(df.head())
print(df['rate'].head(10))
print(df['approx_cost(for two people)'].unique()[:20])
