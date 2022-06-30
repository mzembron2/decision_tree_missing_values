###################################################
#   Author: Mateusz Zembron                       #
###################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from missing_values_creator import MissingValuesCreator
from dt_missing_values import DecisionTreeWithMissingValuesClassifier
import os

######## script with unit tests  - run `pytest` in terminal (in scripts directory)

def test_percentage_of_missing_values_in_data():
    ''' test to check percent of missing values in given data created by MissingValuesCreator class  '''
    
    # loading data
    dirname = os.path.dirname(__file__)
    filename_iris = os.path.join(dirname, '../data/iris.csv')
    data = pd.read_csv(filename_iris, skiprows=1, header=None)
    X = data.iloc[:, :-1].values

    percentage = 65
    index = 1
    
    missing_values_creator = MissingValuesCreator(percentage)
    X_missing_values = missing_values_creator.add_random_missing_values_by_list(X, [index])
    number_of_elements_missing = 0
    
    # it was easier to loop than use count() method - "nan" has some stange type
    for element in X_missing_values[:, 1].tolist():
        if str(element) == "nan":
            number_of_elements_missing += 1

    number_of_all_elements = len(X_missing_values[:, 1].tolist())
    assert(round(number_of_elements_missing/number_of_all_elements,2) == (percentage/100))

def test_prediction_of_element_with_missing_value():

    ''' test to validate prediction of element's value on iris dataset  '''
    dirname = os.path.dirname(__file__)
    filename_iris = os.path.join(dirname, '../data/iris.csv')

    col_names_iris = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
    data = pd.read_csv(filename_iris, skiprows=1, header=None, names=col_names_iris)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=20) # random_state is a seed
    # with known seed split can be reproduced 
    classifier = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf,  missing_values_predictor= 1) #there is no constraint on decision tree depth
    classifierNaive = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf,  missing_values_predictor= 2) #there is no constraint on decision tree depth
    
    # structure of tree can be shown by: `classifier.print_tree()`
    classifier.fit(X_train,Y_train) # test given dataset
    classifierNaive.fit(X_train,Y_train)
    # with known structure of decision tree, i can predict output of classifier 
    test1 = [[6.1, 2.8, "nan", 1.2]] #it is 'Versicolor' and classifier should predict it like that (manual calculations)
    test2 = [[4.6, 3.2, "nan", 0.2]] #it is 'Setosa' and classifier should predict 'Versicolor'(manual calculations)
    test3 = [[6.5, 3.0, "nan", "nan"]] #it is Virginica and classifier should predict 'Setosa' (even tho naive classifier should predict it properly - manual calculations)
                                       #this is an example when naive classifier classifies better than normal one 


    assert(classifier.predict(test1)[0] == 'Versicolor')
    assert(classifier.predict(test2)[0] == 'Versicolor')
    assert(classifier.predict(test3)[0] == 'Setosa') # but naive:
    assert(classifierNaive.predict(test3)[0] == 'Virginica')


    



