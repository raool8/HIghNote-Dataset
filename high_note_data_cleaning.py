# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:23:35 2019

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
dataset = pd.read_csv("curated_High_note_data.csv", na_values = missing_values)
X = dataset.iloc[:, [1,3]].values
y = dataset.iloc[:, 32].values

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
#Having estimated the age of the net user, delta1_avg_friend_age and delta2_avg_friend_age is inconsequential to the analysis, hence dropping
dataset.drop(['delta1_avg_friend_age','delta2_avg_friend_age','delta1_avg_friend_male','delta2_avg_friend_male'], axis = 1, inplace = True)
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
dataset['delta1_good_country'].replace(-1,1, inplace = True)
dataset['delta2_good_country'].replace(-1,1, inplace = True)
#Determining the number of good_country and otherwise users 
dataset.groupby('good_country').count()
dataset.groupby('delta1_good_country').count()
dataset.groupby('delta2_good_country').count()
#Since, from the above 3 lines of code tell us that 65701 and 66141 users stayed in a good country for the corresponding 6 months and 23994 users stay in the good country in the current period,
#We can safely assume that the current inhabitants of good country and the ones that stayed in the adjoining 6 month period must be people who stay in the good country for long periods.
#dataset['good_country'] = dataset.apply(
        #lambda row: row['good_country'] == 1  if np.isnan(row['delta1_good_country'] == 0 or row['delta2_good_country'] == 0) else row['good_country'] == 0, axis = 1
        #)
dataset['good_country'].fillna(dataset['delta2_good_country'], inplace = True)
dataset['good_country'].fillna(dataset['delta1_good_country'], inplace = True)
#Assuming there is equal represenation of travelling and indigenous users within the current period among the missing values in the good country column 
#Hence, #locals = #travellers = 12406
dataset['good_country'].fillna(0, limit = 12406, inplace = True)
dataset['good_country'].fillna(1, limit = 12405, inplace = True)
dataset['delta1_good_country'].fillna(dataset['good_country'], inplace = True)
dataset['delta2_good_country'].fillna(dataset['good_country'], inplace = True)

#Replacign all missing values in the remaining columns with their respective mean values, since the range of missing values ranges from 8 - 1865(upto 2% of the whole dataset)
dataset['shouts'].fillna(dataset['shouts'].mean(), inplace = True)
dataset['delta1_friend_cnt'].fillna(dataset['delta1_friend_cnt'].mean(), inplace = True)
dataset['delta1_friend_country_cnt'].fillna(dataset['delta1_friend_country_cnt'].mean(), inplace = True)
dataset['delta1_subscriber_friend_cnt'].fillna(dataset['delta1_subscriber_friend_cnt'].mean(), inplace = True)
dataset['delta1_songsListened'].fillna(dataset['delta1_songsListened'].mean(), inplace = True)
dataset['delta1_lovedTracks'].fillna(dataset['delta1_lovedTracks'].mean(), inplace = True)
dataset['delta1_posts'].fillna(dataset['delta1_posts'].mean(), inplace = True)
dataset['delta1_playlists'].fillna(dataset['delta1_playlists'].mean(), inplace = True)
dataset['delta1_shouts'].fillna(dataset['delta1_shouts'].mean(), inplace = True)
dataset['tenure'].fillna(dataset['tenure'].mean(), inplace = True)
dataset['delta2_friend_cnt'].fillna(dataset['delta2_friend_cnt'].mean(), inplace = True)
dataset['delta2_friend_country_cnt'].fillna(dataset['delta2_friend_country_cnt'].mean(), inplace = True)
dataset['delta2_subscriber_friend_cnt'].fillna(dataset['delta2_subscriber_friend_cnt'].mean(), inplace = True)
dataset['delta2_songsListened'].fillna(dataset['delta2_songsListened'].mean(), inplace = True)
dataset['delta2_lovedTracks'].fillna(dataset['delta2_lovedTracks'].mean(), inplace = True)
dataset['delta2_posts'].fillna(dataset['delta2_posts'].mean(), inplace = True)
dataset['delta2_playlists'].fillna(dataset['delta2_playlists'].mean(), inplace = True)

#Exporting curated CSV file 
export_csv = dataset.to_csv (r'D:\Data Visualization Projects\High note data\curated_High_note_data.csv', index = False, header=True)




#Useful code for preliminary analysis
plt.hist(dataset.friend_cnt, alpha = 1)
plt.show()

max(dataset.age)
min(dataset.songsListened)

dataset.loc[dataset['adopter'] == 1,'net_user'].sum()
e= dataset.groupby('good_country').mean()
dataset.groupby('subscriber_friend_cnt').sum().where(dataset['subscriber_friend_cnt'] <= 50)
dataset['age'].min()
dataset['delta2_good_country'].unique()
dataset.dtypes
dataset.describe()
dataset['age'].mean()
dataset['delta1_avg_friend_male'].mean()
dataset['age'].min()
dataset['friend_country_cnt'].mean().where(dataset['good_country'] == 1)