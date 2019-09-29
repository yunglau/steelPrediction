import pandas as pd

df1 = pd.read_csv("historical_country_united_states_indicator_inflation_rate.csv")
del df1['Country']
del df1['Category']
del df1['HistoricalDataSymbol']
del df1['Frequency']
del df1['LastUpdate']

df1['DateTime'] = pd.to_datetime(df1['DateTime'], format='%Y-%m-%d')
dates = pd.date_range(start='2009-06-30', end='2019-09-27', name='DateTime')
dfIron_Ores_date = dates.to_frame(index=False)
dfIron_Ores_date.columns = ['DateTime']
df = pd.merge(dfIron_Ores_date, df1, on=['DateTime'], how='outer')

df = df.sort_values(by=["DateTime"], ascending=True)
df = df.ffill()
df.reset_index(level=0, inplace=True)
del df['index']
export_csv = df.to_csv('US_Inflation.csv', index = None, header=True)
print(df)
