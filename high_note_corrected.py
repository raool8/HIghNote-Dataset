# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:30:34 2019

@author: Rahul
"""

'''Objectives:
    1) How can we convert more free customers to paid subscription holders?
    2) How can we attract more premium subscribers?
    3) How can we keep the users we have engaged without having them wander off? '''

#Importing the libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

#Importing the dataset
missing_values = ["na","NA","Na"," ","--","nan"] 
dataset = pd.read_csv("High Note data csv.csv", na_values = missing_values)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 37].values

#Checking for missing values 
dataset.isnull().sum()
dataset.drop('delta2_shouts', axis = 1, inplace = True) #All data missing
#dataset.drop(['delta1_avg_friend_age', 'delta2_avg_friend_age'], axis = 1, inplace = True)#Current age has been deduced using these columns, hence no longer required. 

#Considering that the % of adopters is quite small already, deleting adopters is counter-intuitive. Hence, non-adopters and nans from all age columns consititue good amount of data that can be deleted without losing the desired results.
dataset.drop(dataset[(dataset['adopter'] == 0) & (pd.isna(dataset['delta2_avg_friend_age'])) & (pd.isna(dataset['delta1_avg_friend_age'])) & (pd.isna(dataset['age']) & (pd.isna(dataset['avg_friend_age'])))].index, inplace = True )

#Using avg_friend_age from current, pre and post periods is a good idea to estimate the age of the user.(Verified on Tableau in preliminary analysis)
dataset['age'].fillna(dataset['avg_friend_age'], inplace = True)
dataset['avg_friend_age'].fillna(dataset['age'], inplace = True) #Having filled out the age column, we can use these values to fill in the estimated avg_friend ages. 
#With only 1062 missing values(1% of the whole dataset) in the age column remaining, we can assume the mean value of the column for these remaining missing spaces. 
#mean age = 24.74 ~ 25 years 
dataset['age'].fillna(25, inplace = True)
#Filling male missing values in the ratio of 1.6 (assumption based on preliminary analysis)
#Hence, 24980 missing places can be filled in as 15370 males and 9610 females 
dataset['male'].fillna(1, limit = 15370, inplace = True) 
dataset['male'].fillna(0, inplace = True)
dataset['avg_friend_male'].fillna(dataset['avg_friend_male'].mean(), inplace = True)

#Cleaning and organising the data 
dataset.sort_values('net_user',inplace = True) 
#In order to estimate the number of missing males and females among net users, it is necessary to understand the relationship between genders among users and their friends.
#Total males = 41,212 and Total females = 24,962 ~ ratio 1.6
#For analysing whether males and females make more male/ female friends:
dataset.groupby((dataset['male'] == 0) & (dataset['avg_friend_male'] == 0.5)).count()
dataset.groupby((dataset['male'] == 1) & (dataset['avg_friend_male'] == 0.5)).count()
dataset.groupby((dataset['male'] == 0) & (dataset['avg_friend_male'] > 0.5)).count()
dataset.groupby((dataset['male'] == 1) & (dataset['avg_friend_male'] > 0.5)).count()
dataset.groupby((dataset['male'] == 0) & (dataset['avg_friend_male'] < 0.5)).count()
dataset.groupby((dataset['male'] == 1) & (dataset['avg_friend_male'] < 0.5)).count()
#Since both genders tend to make more male friends(owing to a greate population), we assume the missing values to be divided to maintain the same ratio (1.6)

#Converting delta1_good_country and delta2_good_country into a two category column (locals/ immigrants) as opposed to 3
dataset['delta1_good_country'].replace(-1,0, inplace = True)
dataset['delta2_good_country'].replace(-1,0, inplace = True)
#Determining the number of good_country and otherwise users 
dataset.groupby('good_country').count()
dataset.groupby('delta1_good_country').count()
dataset.groupby('delta2_good_country').count()
#Since, from the above 3 lines of code tell us that 65701 and 66141 users stayed in a good country for the corresponding 6 months and 23994 users stay in the good country in the current period,
#We can safely assume that the current inhabitants of good country and the ones that stayed in the adjoining 6 month period must be people who stay in the good country for long periods.
dataset['good_country'] = dataset.apply(
        lambda row: row['delta1_good_country'] == 0 or row['delta2_good_country'] == 0 if np.isnan(row['good_country'] == 1) else row['good_country'] == 0, axis = 1
        )

dataset.describe()

plt.hist(dataset.friend_cnt, alpha = 1)
plt.show()

max(dataset.age)
min(dataset.songsListened)

dataset.loc[dataset['adopter'] == 1,'net_user'].sum()
dataset.groupby('delta2_good_country').count()
dataset.groupby('adopter').sum()
dataset['subscriber_friend_cnt'].sum()
dataset['delta2_good_country'].unique()
dataset.dtypes
dataset.describe()
dataset['age'].mean()
dataset['delta1_avg_friend_male'].mean()
dataset['age'].min()