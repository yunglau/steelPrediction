import pandas as pd

df_Oil = pd.read_csv("Crude Oil prices.csv")
df_Oil['Date'] = pd.to_datetime(df_Oil['Date'], format='%d/%m/%Y')
dates = pd.date_range(start='28/09/2009', end='27/09/2019', name='Date')
dfIron_Ores_date = dates.to_frame(index=False)

df = pd.merge(dfIron_Ores_date, df_Oil, on=['Date'], how='outer')

df = df.sort_values(by=["Date"], ascending=True)
df = df.ffill()
export_csv = df.to_csv ('Crude_Oil.csv', index = None, header=True)
print(df)
