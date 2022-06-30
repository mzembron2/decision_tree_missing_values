###################################################
#   Author: Mateusz Zembron                       #
###################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from missing_values_creator import MissingValuesCreator
from dt_missing_values import DecisionTreeWithMissingValuesClassifier
from random import randrange
import os
''' File for main experiments '''


# path to data
dirname = os.path.dirname(__file__)
filename_iris = os.path.join(dirname, '../data/iris.csv')
filename_iris_reduced = os.path.join(dirname, '../data/iris_reduced.csv')
filename_wine = os.path.join(dirname, '../data/wine.csv')
filename_cancer = os.path.join(dirname, '../data/breast_cancer.csv')


# data handling 

''' uncoment any data you are willing to use  '''

''' irises '''
# col_names_iris = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
# data = pd.read_csv(filename_iris, skiprows=1, header=None, names=col_names_iris)

''' irises reduced to 18 examples '''
# dataset created for checking the algorithm

# col_names_iris = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
# data = pd.read_csv(filename_iris_reduced, skiprows=1, header=None, names=col_names_iris)

''' wines '''
col_names_wine = ["type","alcohol","malic.acid","ash","alcalinity.of.ash","magnesium","total.phenols","flavanoids","nonflavanoid.phenols","proanthocyanins","color.intensity","hue","OD280/OD315.of.diluted.wines","proline"]
data = pd.read_csv(filename_wine, skiprows=1, header=None, sep = ';',names=col_names_wine)
# reconfiguring order of columns - values of elements as last column 
data = data[["alcohol","malic.acid","ash","alcalinity.of.ash","magnesium","total.phenols","flavanoids","nonflavanoid.phenols","proanthocyanins","color.intensity","hue","OD280/OD315.of.diluted.wines","proline","type"]]

''' breast cancer '''
# data = pd.read_csv(filename_cancer, skiprows=1, header=None, sep = ';')
# # reconfiguring order of columns - values of elements as last column 
# data_order = np.linspace(2, 31, num = 30, dtype = int).tolist() + [1]
# data = data[data_order]



X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)


seed = randrange(1, 1000)
number_of_iterations = 50
list_of_atributes_missing_values = [6]
avg_accuracy_full_classifier = 0
avg_accuracy_naive_classifier = 0
avg_accuracy_classifier_without_missing_values = 0





for n in range(number_of_iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=seed)# random state is a seed,

    classifier = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf,  missing_values_predictor= 1) #there is no constraint on decision tree depth
    classifier.fit(X_train,Y_train)
    # classifier.print_tree()

    full_classifier = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf,  missing_values_predictor= 1) #there is no constraint on decision tree depth
    full_classifier.fit(X_train,Y_train)

    naive_classifier = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf, missing_values_predictor= 2) #naive probabilistic approach to missing values 
    naive_classifier.fit(X_train,Y_train)

    Y_without_missing_values = classifier.predict(X_test) #normal classification of full data without missing values 

    avg_accuracy_classifier_without_missing_values  += accuracy_score(Y_test, Y_without_missing_values)

    missing_values_creator = MissingValuesCreator()
    X_test_missing = missing_values_creator.add_missing_values_by_list(X_test, list_of_atributes_missing_values)

    Y_pred_full = full_classifier.predict(X_test_missing) 
    Y_pred_naive = naive_classifier.predict(X_test_missing) 

    avg_accuracy_full_classifier += accuracy_score(Y_test, Y_pred_full)
    avg_accuracy_naive_classifier += accuracy_score(Y_test, Y_pred_naive)

    # print(accuracy_score(Y_test, Y_pred_full))
    # print( accuracy_score(Y_test, Y_pred_naive))

    # print(avg_accuracy_classifier_without_missing_values)
    # print(avg_accuracy_full_classifier)
    # print(avg_accuracy_naive_classifier)
    seed = randrange(1, 1000) #new seed


avg_accuracy_full_classifier = avg_accuracy_full_classifier/number_of_iterations
avg_accuracy_naive_classifier = avg_accuracy_naive_classifier/number_of_iterations
avg_accuracy_classifier_without_missing_values = avg_accuracy_classifier_without_missing_values/number_of_iterations

print("Classification without missing values: ",avg_accuracy_classifier_without_missing_values)
print("Classification with full probabilistic approach: ", avg_accuracy_full_classifier)
print("Classification with naive probabilistic approach: ", avg_accuracy_naive_classifier)


    
