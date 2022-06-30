###################################################
#   Author: Mateusz Zembron                       #
#   Author: Daniel Adamkowski                     #
###################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dt_missing_values import DecisionTreeWithMissingValuesClassifier
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree



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
# # # reconfiguring order of columns - values of elements as last column 
# data_order = np.linspace(2, 31, num = 30, dtype = int).tolist() + [1]
# data = data[data_order]


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
# # change of random state variable genereates new order of train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=80) 


''' implemented algorithm'''
classifier = DecisionTreeWithMissingValuesClassifier(min_samples_split=3, max_depth=np.inf,  missing_values_predictor= 1) #there is no constraint on decision tree depth
classifier.fit(X_train,Y_train)
classifier.print_tree()


''' sklearn algorithm '''
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model = clf.fit(X_train, Y_train)

text_representation = tree.export_text(clf)
print(text_representation)


Y_pred = classifier.predict(X_test) 
Y_pred_ref = model.predict(X_test)
print("Dokladnosc mojego modelu: ", accuracy_score(Y_test, Y_pred))
print("Dokladnosc modelu sklearn : ", accuracy_score(Y_test, Y_pred_ref))
