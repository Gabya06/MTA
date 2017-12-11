# MTA data science project

## Project Overview
The goal of this project is to scrape the MTA website for data, perform analysis on the number of turnstyle entries and exits and predict the number of turnstyle entries and exitis.

## Data Scrapping and Processing 
MTA turstile data was scrapped from http://web.mta.info/developers/turnstile.html. There were two different data formats used, one for data before 10/8/2014 and one for after. In my analysis, I named the dataset containing the old format old_data and the new format new_data. I created a function to iteratively read the online data and add it to a text file. At the end of the data scrapping phase there are two text files that were created, one for each type of format.
After reading the data, I cleaned up the format of the old data to match the new one. Since there is a lot of data, processing this took quite some time. In hindsight, I should have created a sqlite database and manipulated the data there, and done some of the analysis there. After some time, I realized that there was a bit too much data to try and combine both the new and old datasets into one pandas dataframe. Since I had already done most of the work in pandas at that point, I focused most of my analysis on the new data to reduce the volume.

```python
def combine_urls(date_range,out_file,mta_data_type):
    '''
    Function to write all text from urls into file
    
    :param date_range: pd.date_range object 
    :param out_file: path + name of file to write to
    :param mta_data_type: 'new' or 'old' to distinguish data 
                        before 10/18/2014 which was formatted differently
    :return: .txt file
    '''
    base_url = 'http://web.mta.info/developers/data/nyct/turnstile/'
    error_list = []
    for i in date_range:
        s = datetime.datetime.strftime(i, '%y%m%d')
        url = base_url + 'turnstile_' + s + '.txt'
        r = requests.get(url=url)
        if r.status_code==404:
            pass
        else:
            content = r.content.decode('utf-8')
            # new formatted data has header that needs to be removed
            if mta_data_type =='new':
                content_header = content.strip().split('\n')[0]
                content = content.strip(content_header)
                
            try:
                with open(out_file, 'a') as outfile:
                    outfile.write(content)
                    #print "wrote ", url, " content to file"
            except Exception as e:
                print "error" ,e
                error_list.append(i)
    return outfile
```

## Data Cleaning
Data cleaning involving standardizing the old and new data to the same format. The old data dating before 10/18/2014 was in a long format, so I change the format to match that of the new data. I reformated the Description, Date and Time columns, and combined all the Entries and Exits into one column, and created a turnstile busyness metric. This metric is the combination of total Entries + Exits. Also, I added the Station and LineName columns based on the mapping file online.

```python
# fill null values in numeric columns (entries & exists)
entries = old_data.columns[old_data.columns.str.contains('ENTRIES')]
old_data[entries] = old_data[entries].fillna(0.0)
exits = old_data.columns[old_data.columns.str.contains('EXITS')]
old_data[exits] = old_data[exits].fillna(0.0)
# create column of total entries and exists
old_data['ENTRIES'] = old_data[entries].sum(axis=1)
old_data['EXITS'] = old_data[exits].sum(axis=1)
old_data = old_data.drop(entries + exits, axis=1)
# combine DESC
desc = old_data.columns[old_data.columns.str.contains('DESC')]
temp = pd.melt(frame=old_data, value_name='DESC', value_vars= desc.values.tolist())
old_data['DESC'] = temp['DESC']
old_data = old_data.drop(desc, axis = 1)
# combine DATE
dates = old_data.columns[old_data.columns.str.contains('DATE')]
temp = pd.melt(frame=old_data,value_name='DATE', value_vars=dates.values.tolist())
old_data['DATE'] = temp['DATE']
old_data = old_data.drop(dates, axis = 1)
# combine TIME
times = old_data.columns[old_data.columns.str.contains('TIME')]
temp = pd.melt(frame=old_data,value_name='TIME', value_vars= times.values.tolist())
old_data['TIME'] = temp['TIME']
old_data = old_data.drop(times, axis = 1)
```

## Data Exploration & Visualization

_4th of July 2016 - what station was busyest?_
42nd street Port Authority had the most turnstile activity, followed by 57th st 7th ave, 23rd st, Canal st and 125th street; and, in particular Unit R011 was busyest.
```python
# 42 ST-PORT AUTH was the busyest around 4th of july 2016
print "ON 4TH OF JULY 2017, THE BUSYEST STATIONS WERE: " 
print
print new_data[new_data.DATE == '07/04/2016'].groupby('STATION').turnstile_busyness.sum().nlargest(5)
```

ON 4TH OF JULY 2017, THE BUSYEST STATIONS WERE: 

| Station            | Turnstile busyness |
|------------------- |:--------------:|
|  42 ST-PORT AUTH   | 8.514554e+10   |
|    57 ST-7 AV      | 8.209046e+10   |
|    23 ST           | 7.544203e+10   |
|    CANAL ST        | 7.186917e+10   |
|    125 ST          | 6.517888e+10   |


_Turnstile busyness by month_
In order to be able to look at the data on a monthly basis, we need to convert datatypes from strings to dates. From the chart, we can see that activity has increased in the last 3 years, but there are a few months where there are dips in turnstile busyness.

```python
# convert date from string
new_data.DATE = new_data.DATE.map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
# add month
new_data['MONTH'] = new_data.DATE.map(lambda x: x.month)
# add year
new_data['YEAR'] = new_data.DATE.map(lambda x: x.year)
# turnstile busyness on any given month
turnstile_yearly = pd.DataFrame(new_data.groupby(['YEAR','MONTH'])['turnstile_busyness'].sum())
# plot yearly data for the last 3 years
turnstile_yearly.plot(kind = 'bar', figsize = (12,6), title = 'Yearly turnstile busyness')
```
