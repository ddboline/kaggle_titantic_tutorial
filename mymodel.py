#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd

def preprocess_data(dataframe):
    dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    if len(dataframe.Embarked[ dataframe.Embarked.isnull() ]) > 0:
        dataframe.Embarked[ dataframe.Embarked.isnull() ] = dataframe.Embarked.dropna().mode().values


    Ports = list(enumerate(np.unique(dataframe['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = dataframe['Age'].dropna().median()
    if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:
        dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


def mymodel():
    traindf = pd.read_csv('train.csv')
    # testdf = pd.read_csv('test.csv')
    
    traindf = preprocess_data(traindf)
    # testdf = preprocess_data(testdf)
    
    print(traindf.columns)
    print(traindf.describe())
    # print(traindf.head())
    # print(testdf.head())
    
    # print(traindf['Sex'][traindf['Survived'] == 1][traindf['Sex'] == 'male'].count())
    # print(traindf['Survived'].sum(), traindf['Survived'].count())
    
    list_of_men = traindf['Sex'] == 'male'
    list_of_women = traindf['Sex'] == 'female'
    list_of_survivors = traindf['Survived'] == 1
    
    # Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age', u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'], dtype='object')
    
    vars_to_consider = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for var in vars_to_consider:
        
    
    
    return

if __name__ == '__main__':
    mymodel()