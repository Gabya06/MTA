# MTA data science project

## Project Overview
The goal of this project is to scrape the MTA website for data, perform analysis on the number of turnstyle entries and exits and predict the number of turnstyle exits.

## Data Scrapping and Processing 
MTA turstile data from 2011 to 2017 was scrapped from http://web.mta.info/developers/turnstile.html. There were two different data formats used, one for data prior to 10/8/2014 and one for after. I created a function to iteratively read the online data and add it to a text file. At the end of the data scrapping phase there are two text files that were created, one for each type of format.
Part of the data cleaning was to standardize both dataset formats to match that of the new data since it was cleaner. The datasets were both large, data prior to 2014 is made up of over 5.7M rows (5,771,751) and data after 2014 had over 2.7M rows (2,652,245). Processing the data took quite some time. In hindsight, I should have made this a lot more efficient by creating  a database, manipulated and done some of the analysis there. After some time, I realized that there was a bit too much data to try and combine both the new and old datasets into one dataframe. Since I had already done most of the work in python using pandas, I focused most of my analysis on the new data to reduce the volume.

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

```python
# 42 ST-PORT AUTH was the busyest around 4th of july 2016
print new_data[new_data.DATE == '07/04/2016'].groupby('STATION').turnstile_busyness.sum().nlargest(5)
```

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
![turnstyle_img](/images/turnstyle_img.png)

_When are turstiles least active in the last year?_

If we look at the data from the last 3 years on a monthly basis, turnstile busyness has increased, but there are certainly also dips in the trend.
In terms of least activity in the last year, we can see that Path WTC has the least turnstyle busyness actvitiy, and February was the least busyest month. The Path WTC station has been closed and under repair for some time, so this would explain the lack of traffic there. It looks like June has the least turnstile business, but this is because the data collected is not quite complete for June, so we should look at the following month with smallest number of exits and entries.
In looking at the stations with turnstile busyness falling in the bottom 5%, we can see that there has been decreasing activity everywhere. In particular, the station with the highest decrease in activty in 2017 was Spring street, followed by other stations that also showed a percentage decrease in usage. Also, there are many stations not operating to full capacity such as: WENTY THIRD ST, PATH WTC, 14TH STREET, FLUSHING AV, NEWARK HM HE,LACKAWANNA,47-50 STS ROCK,CANARSIE-ROCKAW and ROOSEVELT ISLND. To visualize this, I sampled a few stations and produced plots of yearly trends by station and noticed that there are many stations that are not used to capacity in comparison with other more popular ones. In terms of least busyest days, there are many days in January which were the least busy, but the day that had the fewest entries and exists was in March.

```python
# in looking at the last 3 years turnstile busyness has been pretty consistent
fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (10,6))
pd.DataFrame(new_data.groupby(['YEAR','MONTH'])['turnstile_busyness'].sum()).plot(kind = 'bar', cmap='cool', ax=ax,
title='Turnstile Busyness in the last 3 years')
plt.show()
```
![turnstyle_img_yearmonth](/images/turnstyle_img_yearmonth.png)


```python
fig, axes = plt.subplots(nrows =1, ncols=4, figsize = (16,6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=.8, wspace=.5)
for idx, s in enumerate(sample_stations[0:4]):
    station_dat = new_data[new_data.STATION ==s].groupby(['MONTH'])
    station_dat = station_dat['turnstile_busyness'].sum()
    plt_s = station_dat.plot(kind = 'bar', subplots=False, stacked = True, legend=None, color ='coral',
                             title = s, ax = axes[idx] )

    labels = [item.get_text() for item in axes[idx].get_xticklabels()]
    axes[idx].set_xticklabels(labels, rotation = 0)
    axes[idx].set_xlabel("")
plt.show()

fig, axes = plt.subplots(nrows =1, ncols=4, figsize = (16,6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=.8, wspace=.75)
for idx, s in enumerate(sample_stations[4:8]):
    station_dat = new_data[new_data.STATION ==s].groupby(['MONTH'])
    station_dat = station_dat['turnstile_busyness'].sum()
    plt_s = station_dat.plot(kind = 'bar', subplots=False, stacked = True, legend=None, color ='coral',
                             title = s , ax = axes[idx] )

    labels = [item.get_text() for item in axes[idx].get_xticklabels()]
    axes[idx].set_xticklabels(labels, rotation = 0)
    axes[idx].set_xlabel("")
plt.show()
```
![turnstyle_samples](/images/turnstyle_samples.png)
![turnstyle_samples_2](/images/turnstyle_samples_2.png)

## Linear Regression Model
I used linear regression to predict the number of exists. While only creating one training and test split of 90% and 10% data, the linear model performed surprisingly quite well without overfitting, scoring 90% on test data and close to 85% on training data. Performing 10 fold cross-validation gives an average score of 84%.

```python
# assign predictors and response
X = model_data.drop(['EXITS'],axis=1)
y = model_data.EXITS

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(scale(X), y, test_size=0.10, random_state=1)
lm = LinearRegression()
# fit linear regression model
lm.fit(X_train, y_train)
print "Linear Regression Model Score on test set",  lm.score(X_test, y_test)
```
Linear Regression Model Score on test set 0.903015614972
 
 To produce the cross-validation plot:
 ```python
 title = "Linear Regression 10-fold cross validation"
# Cross validation with 100 iterations to get smoother mean test and train score curves, 
# each with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n =4, test_size=.2, random_state=0, n_iter=100)

estimator = make_pipeline(RobustScaler(), LinearRegression())
_ = plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=10, n_jobs=-1)
 ```
 ![cv_learningcurves](/images/cv_learningcurves.png)

Note: I found this helpful function to produce the plot for cross-validation:
```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
 ```
