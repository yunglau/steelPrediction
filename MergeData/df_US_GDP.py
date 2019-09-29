import pandas as pd

#Read csv files
df_US_GDP = pd.read_csv("/Users/elaine/PycharmProjects/ImplementAI/historical_country_united_states_indicator_gdp_growth_rate(1).csv")

#Set dataframe 'DateTime' into datetime mode
df_US_GDP['DateTime'] = pd.to_datetime(df_US_GDP['DateTime'], format='%Y-%m-%d')

#Convert the dataframe from monthly mode to daily mode
dates = pd.date_range(start='2009-06-30', end='2019-09-27', name='DateTime')
dfIron_Ores_date = dates.to_frame(index=False)
dfIron_Ores_date.columns = ['DateTime']
df = pd.merge(dfIron_Ores_date, df_US_GDP, on=['DateTime'], how='outer')

#Fill the gaps with the previous slots
df = df.sort_values(by=["DateTime"], ascending=True)
df = df.ffill()

#Reset the index and delete the initial index placed
df.reset_index(level=0, inplace=True)
del df['index']

print(df)
#export the csv file
export_csv = df.to_csv('US_GDP.csv', index = None, header=True)
