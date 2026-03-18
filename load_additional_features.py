import pandas as pd
import numpy as np

# Load from file
df2023 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2023')
df2023['YEAR'] = 2023

df2024 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2024')
df2024['YEAR'] = 2024

df2025 = pd.read_excel('NCAA_Validation_Additional_Features.xlsx', sheet_name='2025')
df2025['YEAR'] = 2025

df = pd.concat([df2023, df2024, df2025])

# Format columns, then drop
df['Last Before 64'] = pd.to_datetime(df['Last Before 64'])
df['64 Date'] = pd.to_datetime(df['64 Date'])

df['REST_DAYS'] = np.where(df['Play In'], 1, (df['64 Date'] - df['Last Before 64']).dt.days)
df['CT_WIN_PCT'] = df['Wins Conf'] / (df['Wins Conf'] + df['Losses Conf'])
df['GAMES_LAST_WEEK'] = df['Conf Games Last Week'] + df['Play In']

df = df.rename(columns={'Wins Conf': 'CT_WIN_TOTAL'})
df = df[['Team', 'YEAR', 'REST_DAYS', 'CT_WIN_PCT', 'GAMES_LAST_WEEK', 'CT_WIN_TOTAL']]
print(df)