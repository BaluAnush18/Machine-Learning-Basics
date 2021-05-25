import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#REGRESSION IS TO PREDICT A REAL CONTINUOUS REAL VALUE.
#CLASSIFICATION IS TO PREDICT A CATEGORY OR A CLASS.

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
# x_train -> Independent variable  | y_train -. dependent variable.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #to create a instance
regressor.fit(x_train, y_train) #used to train the model on the training set or predicts the future model.

#PREDICTING THE RESULT BASED ON TEST VALUES:
y_pred = regressor.predict(x_test)

#VISUALIZING THE TRAINING SET:

plt.scatter(x_train, y_train, color='red') #input the coordinates.
plt.plot(x_train, regressor.predict(x_train), color='blue') #used to plot the regression line(Coming close to the straight line).
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#RUNNING THE TEST SET:

plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color='orange')
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()