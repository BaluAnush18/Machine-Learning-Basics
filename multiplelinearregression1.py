#SAME AS LINEAR REGRESSION BUT HAS MANY COMBINATIONS OF b & x.
#LINEAR REGRESSION -> y = b0 + b1*x1
#MULTIPLE REGRESSION -> y = b0 + b1*x1 + b2*x2 + b3*x3 + bn*xn

#ASSUMPTIONS OF LINEAR REGRESSION:
#Linearity, Homoscedasticity, Multivariate Normality, Independence of Errors, Lack of Multicollinearity.

#DUMMY VARIABLES:
#In this case, Profit is the dependent variable. So b0 is profit.
#b1*x1 -> R&D Spend ; b2*x2 -> Admin ; b3*x3 -> Marketing ; But problem arises when we encounter a categorical varibles.
#To overcome categorical variable problem, we need to create a dummy variable.
#Eg: For values in NY column, put 1 -> present & 0 elsewhere. Similarly, for California put 1 -> present & 0 where it is absent.
#So regression equation becomes b4*D1 (name of Dummy variable column).

#DUMMY VARIABLE TRAP:
#We cannot include 2 dummy variable at the same time.
#Because we are basically duplicating the variables. This is because D2 = 1 - D1.
#The phenomenon where one or more variables predict another -> Multiple Linearity.
#As a result, the model cannot distinguish between dummy variables and results in dummy variable trap.
#And also we cannot include a constant(b0) and both the dummy variables at the same time in the same equation.(Refer math)

#STASTICAL SIGNIFICANCE:
#H0 : This is a fair coin ; H1 : This is not a fair coin.
#Suppose we flip the coin and get tail continuously, the probablity of getting a tail everytime is 0.5 -> 0.25 -> 0.12 -> 0.06 -> 0.03 -> 0.01.
#Suppose we do this for 33 days, there is a rare chance of us getting the above combination.
#We are assuming the hypothesis is true in the given universe.
#The combination is called P-Value.
#We get the P-Value in H0 is 0.5 -> 0.25 -> 0.12 -> 0.06 -> 0.03 -> 0.01.
#But in H1 the values would be 100%.
#We assume that we are getting an uneasy feeling and feeling suspicious about out model. This value is alpha and we assume it be 0.05 in our case.
#Once the value goes below alpha, it is unlikely to see this random and it is unlikely to happen, it is right to reject that hypothesis.
#P-Value depends on experiment and results. Ideally it is set to 95%.

#BUILDING A MODEL:
#5 methods: All-in, Backward Elimination, Forward Selection, Bi-directional Elimination, Score Comparison.
#Step wise regression -> Backward Elimination, Forward Selection, Bi-directional Elimination. (default : Bi-directional Elimination).

#All-in : To let all the variables in once you are sure that all the variables are true to your knowledge.

#Backward Elimination :
# Steps ->
# 1. Select a significance level to stay in the model.
# 2. Fit the full model with all possible predictors.
# 3. Consider the predictor with highest P-Value. If P > SL -> Step 4. Else FIN. (FIN -> Finish. Model is ready)
# 4. Remove the predictor.
# 5. Fit the model without this variable. Repeat till P > SL fails.

#Forward Elimination :
# 1. Select a significance level to enter the model.
# 2. Fit all the simple regression models y ~ xn. Select the one with lowest P-Value.
# 3. Keep this variable and fit all the possible models with one extra predictor added to one(s) you already have.
# 4. Consider the predictor with lowest P-Value. If P < SL, goto step 3, otherwise FIN.(FIN -> Keep a step back).

#Bi-directional Elimination:
# 1. Select a significance level to enter and to stay in the model.
# 2. Perform the next step of Forward Selection. (New variables must have: P < SLENTER to enter).
# 3. Perform all steps of Backward Elimination. (Old variables must have: P < SLSTAY to stay).
# 4. Keep repeating until you cannot eliminate a variable or add. No new variables can enter and no old variables can exit. FIN.

#Score Comparison:
# 1. Select a criterion of goodness and fit.
# 2. Construct all possible regression models.((2^n)-1) total combinations.
# 3. Select the one with best criterion. FIN.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer( transformers= [('encoding', OneHotEncoder(), [3])],remainder='passthrough' )
x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#In Multiple Linear Regression, there is no need to apply a feature scaling as the coefficient terms such as b1, b2 will compensate and come on the same scale.

from sklearn.linear_model import LinearRegression #will by default choose the best P-Model and will return it.
regression=LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#To display the real profits and predicted profits, we use concatenate which concats either vertically or horizontally.
#1->concat horizontally. 0->vertically