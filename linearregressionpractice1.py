import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Book1.csv')
x = dataset.iloc[:, :1]
y = dataset.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('CGPA Predictor - Training Data')
plt.xlabel('Sememster')
plt.ylabel('CGPA')
plt.show()

plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color='orange')
plt.title('CGPA Predictor - Test Data')
plt.xlabel('Sememster')
plt.ylabel('CGPA')
plt.show()