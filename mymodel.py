#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

from sklearn import linear_model
from sklearn import svm, neighbors, svm, grid_search
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def preprocess_data(dataframe):
    dataframe['Gender'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    if len(dataframe.Embarked[ dataframe.Embarked.isnull() ]) > 0:
        dataframe.Embarked[ dataframe.Embarked.isnull() ] = dataframe.Embarked.dropna().mode().values

    if len(dataframe.Fare[ dataframe.Fare.isnull() ]) > 0:
        dataframe.Fare[ dataframe.Fare.isnull() ] = dataframe.Fare.dropna().mode().values

    Ports = list(enumerate(np.unique(dataframe['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = dataframe['Age'].dropna().median()
    if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:
        dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = -1

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',
                                'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 
                                #'Pclass',
                                ], axis=1)

    return dataframe

def create_html_page_of_plots(list_of_plots):
    if not os.path.exists('html'):
        os.makedirs('html')
    os.system('mv *.png html')
    print(list_of_plots)
    with open('html/index.html', 'w') as htmlfile:
        htmlfile.write('<!DOCTYPE html><html><body><div>')
        for plot in list_of_plots:
            htmlfile.write('<p><img src="%s"></p>' % plot)
        htmlfile.write('</div></html></html>')
    if os.path.exists('%s/public_html' % os.getenv('HOME')):
        if not os.path.exists('%s/public_html/titanic_html' % os.getenv('HOME')):
            os.makedirs('%s/public_html/titanic_html' % os.getenv('HOME'))
        os.system('rm %s/public_html/titanic_html/*' % os.getenv('HOME'))
        os.system('cp html/* %s/public_html/titanic_html/' % os.getenv('HOME'))


def plot_vars(df):
    list_of_men = df['Gender'] == 1
    list_of_women = df['Gender'] == 0
    list_of_survivors = df['Survived'] == 1
    list_of_casualties = df['Survived'] == 0
    
    vars_to_consider = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender']
    list_of_plots = []
    for var in vars_to_consider:
        pl.clf()
        df[var][list_of_survivors][list_of_women].hist(histtype='step', bins=50, color='blue', label='Survived')
        df[var][list_of_casualties][list_of_women].hist(histtype='step', bins=50, color='red', label='Died')
        pl.title(var)
        pl.legend(loc='upper left')
        pl.savefig('%s_hist.png' % var)
        list_of_plots.append('%s_hist.png' % var)
    create_html_page_of_plots(list_of_plots)

def score_model(model, xtrain, xtest, ytrain, ytest):
    try:
        model.fit(xtrain, ytrain)
        return model.score(xtest, ytest)
    except:
        print('failure')
        return 0.0

def compare_models(traindata):
    classifier_dict = {
                #'gridCV': clf,
                #'linear_model': linear_model.LogisticRegression(fit_intercept=False,penalty='l1'),
                #'linSVC': svm.LinearSVC(),
                #'kNC5': KNeighborsClassifier(),
                #'kNC6': KNeighborsClassifier(6),
                'SVC': SVC(kernel="linear", C=0.025),
                #'SVC': SVC(kernel='rbf', C=0.025, ),
                #'DT': DecisionTreeClassifier(max_depth=5),
                'RF': RandomForestClassifier(n_estimators=400),
                'Ada': AdaBoostClassifier(),
                'Gauss': GaussianNB(),
                #'LDA': LDA(),
                #'QDA': QDA(),
                #'SVC2': SVC(),
              }

    model_scores = {}
    for name in classifier_dict.keys():
        model_scores[name] = []
    for N in range(1):
        randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
        xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(traindata[0::,1::], traindata[0::,0], test_size=0.4, random_state=randint)
        
        #print('Training...')
        #forest = RandomForestClassifier(n_estimators=200)
        #forest = forest.fit(xtrain, ytrain)
        #print(traindf.columns[1:])
        #print(forest.feature_importances_)
        #print('score:', forest.score(xtest, ytest))
        
        for name, cl in classifier_dict.items():
            model_scores[name].append(score_model(cl, xtrain, xtest, ytrain, ytest))
    for k in model_scores:
        model_scores[k] = sum(model_scores[k]) / len(model_scores[k])
    print('\n'.join('%s %f' % (x[0], x[1]) for x in sorted(model_scores.items(), key=lambda x: x[1])))


def mymodel(do_plots=False, do_comparison=False, do_submission=False):
    traindf = pd.read_csv('train.csv')
    testdf = pd.read_csv('test.csv')
    
    testid = testdf['PassengerId'].values
    
    traindf = preprocess_data(traindf)
    testdf = preprocess_data(testdf)
    
    print(traindf.columns)
    print(traindf.describe())

    train_cols = traindf.columns
    test_cols = testdf.columns
    traindata = traindf.values
    testdata = testdf.values

    if do_plots:
        plot_vars(traindf)

    if do_comparison:
        compare_models(traindata)

    if do_submission:
        xtrain = traindata[0::,1::]
        ytrain = traindata[0::,0]
        xtest = testdata
        
        model = SVC(kernel="linear", C=0.025)
        model.fit(xtrain, ytrain)
        ytest = model.predict(xtest)
        
        submitdf = pd.DataFrame(data={'PassengerId': testid, 'Survived': ytest.astype(int)})
        submitdf.to_csv('submit.csv', index=False)

    return

if __name__ == '__main__':
    mymodel(do_comparison=True)
