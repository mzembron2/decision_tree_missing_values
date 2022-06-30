###################################################
#   Author: Mateusz Zembron                       #
###################################################


import numpy as np
import pandas as pd
from probabilistic_predictor import ProbabilisticPredictor


class Node():
    def __init__(self, feature_index = None, threshold=None, left = None, left_dataset = None ,right=None,right_dataset = None , info_gain=None, value=None, final_leaf_dataset = None):
        #for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        #due to left is a 'function', also store number of elements in each branch
        self.left_dataset = left_dataset
        self.right = right
        self.right_dataset = right_dataset
        self.info_gain =info_gain

        #for leaf node
        self.value = value
        self.final_leaf_dataset = final_leaf_dataset

class DecisionTreeWithMissingValuesClassifier():
    def __init__(self, min_samples_split=2, max_depth=2, missing_values_predictor = 1):
        ''' constructor '''
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # variable missing_values_predictor determines approach to missing values 
        # default value: 1 - probabilistic approach 
        # anything else: naive probabilistic approach 
        # details explained in make_prediction() function 
        self.missing_values_predictor = missing_values_predictor
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1] #matrix dimension
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                # addition - counting elements in each branch 
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, best_split["dataset_left"] ,right_subtree, best_split["dataset_right"], best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value, final_leaf_dataset= Y)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        #sth like: best_split["deafult_value_for_missing_values"] = 
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree, probability = 1, list_of_probabilities = 0):
        ''' function to predict a single data point '''
        ## function uses recurencion - for missing values we need to implement 
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        ## IF feature_value IS NOT MISSING and wanst missing in any of previous tests:
        if not self.is_missing(feature_val):   
            if feature_val<=tree.threshold:
                return self.make_prediction(x, tree.left, probability)
            else:
                return self.make_prediction(x, tree.right, probability)
        ## ELSE 
        else:
            if self.missing_values_predictor == 1:
                #creating dictionary with unique values of elements
                full_dataset = np.concatenate((tree.left_dataset, tree.right_dataset), axis = 0)
                unique_names = np.unique(full_dataset[:,-1])
                unique_names_dict = {}
                for u_name in unique_names:
                    unique_names_dict[u_name] = 0
                
                
                prob_predict = ProbabilisticPredictor(unique_names_dict, tree)
                prob_predict.fill_probabilites(x, tree)
                return prob_predict.return_most_possible()
                
            else:
                left_num_of_elements = len(tree.left_dataset)
                right_num_of_elements = len(tree.right_dataset)
                probability_of_left_branch = probability*left_num_of_elements/(right_num_of_elements + left_num_of_elements)
                probability_of_right_branch = probability*right_num_of_elements/(right_num_of_elements + left_num_of_elements)

                if probability_of_left_branch > probability_of_right_branch:
                    return self.make_prediction(x, tree.left, probability)

                else: 
                    return self.make_prediction(x, tree.right, probability)


    def is_missing(self, feature_val):
        '''' checking if value is not missing'''    
        if str(feature_val) == "NaN" or str(feature_val) == "nan":
            return True
        else:
            return False




