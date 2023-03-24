s# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:40:43 2023

@author: Indra.Narakulla
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# # oversampling
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTENC
# from imblearn.over_sampling import ADASYN

# train test split
from sklearn.model_selection import train_test_split

# model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from xgboost import XGBRFClassifier
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer


# model evaluation & tuning hyperparameter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif



#################################################################################
##### Step 2: Reading data
#################################################################################

print(os.getcwd())

# Reading Data Set and creating Pandas Data Frame
os.chdir("C:\\Users\\indra.narakulla\\OneDrive - Kantar\\Training\\Corporate\\LR2\\C1")

data = pd.read_csv("Sales_Conversion.csv")
type(data)


data.head() # read the first five rows
data.tail() # read the last five rows

data.shape # check out the dimension of the dataset

data.dtypes  # look at the data types for each column

data.columns.values  # return an array of column names
data.columns.values.tolist()  # return a list of column names

data.info()

data.isnull().values.any()  # only want to know if there are any missing values

data.isnull().sum() # knowling number of non-missing values for each variable
data.isnull().sum().sum()

data.describe()

# Dropping columns which are not useful - like ID

#### End PART 1 Chaitanya ####
