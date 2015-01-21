#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


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

    return dataframe

def plot_vars(df):
    list_of_men = df['Gender'] == 1
    list_of_women = df['Gender'] == 0
    list_of_survivors = df['Survived'] == 1
    list_of_casualties = df['Survived'] == 0
    
    vars_to_consider = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender']
    for var in vars_to_consider:
        pl.clf()
        #help(df[var].hist)
        #df[var][list_of_men].hist(histtype='step', bins=50, color='blue')
        #df[var][list_of_women].hist(histtype='step', bins=50, color='red')
        df[var][list_of_survivors].hist(histtype='step', bins=50, color='blue')
        df[var][list_of_casualties].hist(histtype='step', bins=50, color='red')
        pl.savefig('%s_hist.png' % var)


def mymodel():
    traindf = pd.read_csv('train.csv')
    # testdf = pd.read_csv('test.csv')
    
    traindf = preprocess_data(traindf)
    # testdf = preprocess_data(testdf)
    
    print(traindf.describe())

    plot_vars(traindf)
    
    traindata = traindf.values
    
    #for n in range(10):
    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(traindata[0::,1::], traindata[0::,0], test_size=0.5, random_state=0)
    
    print('Training...')
    forest = RandomForestClassifier(n_estimators=200)
    forest = forest.fit(xtrain, ytrain)
    print(traindf.columns[1:])
    print(forest.feature_importances_)
    print('score:', forest.score(xtest, ytest))
    
    return

if __name__ == '__main__':
    mymodel()