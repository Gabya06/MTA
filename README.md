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
Data cleaning involving standardizing  data to the same format. I changed the format of the old data to match that of the new data. The old data from before 10/18/2014 was in a long format, so I used the melt function to reformat these columns into one. I did this for the DESC, DATE, and TIME columns. I combined all the ENTRIES into one column and EXITS also, and created the turnstile busyness metric. I added the Station and LineName columns based on the mapping file online.
