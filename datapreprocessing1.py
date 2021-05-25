#IMPORTING LIBRARIES:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATASET:

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values #, seperates the rows and columns and we can take only the columns. Multiple row copy.
#[: (-> range) , (-> seperator between row & column) :-1 (-> upper bound:lower bound)]
#x=dataset.iloc[:2].values : will only copy the value of first 2 rows.
y = dataset.iloc[:, -1].values #single row copy.

#TO RESLOVE MISSING DATA:

from sklearn.impute import SimpleImputer

imputer = SimpleImputer( missing_values=np.nan, strategy = 'mean')
imputer.fit(x [:, 1:3] )
x[ :, 1:3] = imputer.transform(x[ :, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer( transformers= [('encoding', OneHotEncoder(), [0])],remainder='passthrough' )
#[(kind of transformation , class of encoder which is encoding, indexes of column we want to transfrom)]
#By default, only the specified columns in transformers are transformed and combined in the output, and the non-specified columns are dropped. (default of 'drop').
#By specifying remainder='passthrough', all remaining columns that were not specified in transformers will be automatically passed through.
x=np.array(ct.fit_transform(x))

#DATA ENCODING:
#Used to encode the labels so that it is easy during ANN steps. Encodes in ascending order.
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
y=le.fit_transform(y)

#SPLITTING THE DATASET INTO TEST SET AND TRAINING SET:

#Involves the splitting of dataset into 2 sets where :
#TRAINING SET -> The set of data which is going to train the model.
#TEST SET -> The set of data which is going to test the efficiency or performance of the model on new observations.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#x_train -> training model for x dataset.
#y_train -> training model for y dataset.
#x_test -> testing model for x dataset.
#y_test -> testing model for y dataset.
#(x , y -> arguements passed, test_size -> percentage of allocation of data for test & train models(80% for training and 20% for testing is ideal one,
#random_state -> to have the same result displayed. Since random factors like random splitting is going to happen, so that we get the same split.

#FEATURE SCALING :

#This step is to ensure that all the data we enter is on the same scale of values. Will get the mean and standard deviation feature in order to perform scaling.
#Apply feature scaling after splitting the dataset because we are going to deploy test set. Training set will only train the model.
#But test set isn't going to train the model, but it is only going to test the model.
#If we apply feature scaling before splitting, then the model is going to train on both the test set and training set which is not which we want.
#So it would cause a error as we taking values which we do not want to be trained as well.
#So it is done after splitting to prevent the leakage of information on test set.

#Standardization formula: Used all the time.
# x(stand) = (x- mean(x))/standard deviation(x)

#Normalization formula: Recommended when is the data is in normalized form.
# x(norm) = (x-min(x))/max(x)-min(x)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
