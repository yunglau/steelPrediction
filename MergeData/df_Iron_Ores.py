import pandas as pd
import numpy as np
import calendar

#Read from csv files
dfIron_Ores = pd.read_csv("Iron_Ores.csv")

#dfIron_Ores = dfIron_Ores.set_index('DATE').diff()
dfIron_Ores['amount_diff'] = dfIron_Ores['WPU1011'].sub(dfIron_Ores['WPU1011'].shift()).fillna(0)
#dfIron_Ores.reset_index(level=0, inplace=True)
data = dfIron_Ores['DATE'].tolist()


days_in_month = []
increments_to_apply = []

#Split the YYYY-MM_DD to get the month data
for row in data:
    month = int(row.split("-")[1])
    year = int(row.split("-")[0])
    days_in_month.append(calendar.monthrange(year, month)[1])
dfIron_Ores['Month'] = days_in_month

dfIron_Ores['Increments_to_apply'] = dfIron_Ores['amount_diff']*10

dfIron_Ores['Daily_Increments'] = dfIron_Ores.Month/dfIron_Ores.Increments_to_apply
dfIron_Ores['DATE'] = pd.to_datetime(dfIron_Ores['DATE'], format='%Y-%m-%d')
dfIron_Ores = dfIron_Ores[dfIron_Ores['DATE'] >= '2009-01-01']
dfIron_Ores.reset_index(level=0, inplace=True)
del dfIron_Ores['index']

dates = pd.date_range(start='2009-01-01', end='2019-09-27', name='DATE')
dfIron_Ores_date = dates.to_frame(index=False)

df = pd.merge(dfIron_Ores_date, dfIron_Ores, on=['DATE'], how='outer')

df = df.sort_values(by=["DATE"], ascending=True)
df = df.ffill()
df = df[df['DATE'] >= '2009-09-28']
df.reset_index(level=0, inplace=True)
del df['index']
df = df[df.columns.difference(['amount_diff', 'Month','Increments_to_apply','Daily_Increments','LastUpdate'])]

export_csv = df.to_csv ('Iron_Ores_Arranged.csv', index = None, header=True)

print(df)