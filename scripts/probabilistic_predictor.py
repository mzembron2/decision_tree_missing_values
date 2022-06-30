###################################################
#   Author: Mateusz Zembron                       #
###################################################


import numpy as np
import pandas as pd

class ProbabilisticPredictor:
    def __init__(self, unique_names_probability_dict, tree):
        ''' class for handling probabilistic approach '''
        self.name_probability_dict = unique_names_probability_dict
        self.tree = tree


    def fill_probabilites(self, x, tree, probability = 1):
        ''' recursive function to fill name_probabiliti_dict '''
        if tree.value!=None: 
            # one of possible leaves was reached, adding probability to dictionary
            self.name_probability_dict[tree.value] = self.name_probability_dict[tree.value] + probability
    
        else:
            feature_val = x[tree.feature_index]
            if not self.is_missing(feature_val):
                if feature_val<=tree.threshold:
                    return self.fill_probabilites(x, tree.left, probability)
                else:
                    return self.fill_probabilites(x, tree.right, probability)
            else:
                left_num_of_elements = len(tree.left_dataset)
                right_num_of_elements = len(tree.right_dataset)
                probability_of_left_branch = probability*left_num_of_elements/(right_num_of_elements + left_num_of_elements)
                probability_of_right_branch = probability*right_num_of_elements/(right_num_of_elements + left_num_of_elements)
                self.fill_probabilites(x, tree.left, probability_of_left_branch)
                self.fill_probabilites(x, tree.right, probability_of_right_branch)

    def return_most_possible(self):
        ''' function to find best fitting value for given element '''
        # stating with first name of name_probability_dict as most possible
        # if its not, it will be replaced in  
        most_possible = list(self.name_probability_dict.keys())[0]
        for key, value in  self.name_probability_dict.items():
            if (value>self.name_probability_dict[most_possible]):
                most_possible = key
        return most_possible

    def is_missing(self, feature_val):
        '''' checking if value is not missing'''    
        if str(feature_val) == "NaN" or str(feature_val) == "nan":
            return True
        else:
            return False