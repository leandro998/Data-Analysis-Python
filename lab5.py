# This file contains the module 5 tests runned during the video classes
import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Import clean data
# path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
path = '/Users/leandro998/Downloads/Data Analysis with Python - IBM/Labs/Lab5 - model evaluation and refinement/module_5_auto.csv'
df = pd.read_csv(path)
# print(df.head())

df = df._get_numeric_data()
# print(df.head())

# Functions for plotting:
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 10
    height = 5
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color='r', label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color='b', label=BlueName)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of cars')
    plt.legend()

    plt.show()
    # plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 10
    height = 5
    plt.figure(figsize=(width, height))

    # Training Data
    # Testing Data
    # lr: linear regression object
    # poly_transform: polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predict Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    # plt.close()


# Part 1: train and test

y_data = df['price']
x_data = df.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

lre = LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
lre.score(x_test[['horsepower']], y_test)
lre.score(x_train[['horsepower']], y_train)
# compare R^2:
print('R^2 > test model: ', lre.score(x_test[['horsepower']], y_test), 'train model: ', lre.score(x_train[['horsepower']], y_train))

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.10, random_state=0)
print('number of test: ', x_test1.shape[0])
print('number of training: ', x_train1.shape[0])
print('R^2 > test model: ', lre.score(x_test1[['horsepower']], y_test1), 'train model: ', lre.score(x_train1[['horsepower']], y_train1))

# Cross Validation:
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print('Rcross: ', Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())
print('Negative Squared Error:')
print(-1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error'))

# predict:
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print('Predict yhat: ', yhat[0:5])

# PART 2 - Overfitting, Underfitting and Model Selection:
print('PART 2: ')
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print('yhat_train:', yhat_train[0:5])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print('yhat_test:', yhat_test[0:5])

# Create a chart:
# Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
# DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
# Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)

# Overfitting:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)
# print(yhat[0:5])
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
# Chart for polynomial overfitting:
# PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
print('R^2 of the data: ', poly.score(x_train_pr, y_train))
print('R^2 of the test: ', poly.score(x_test_pr, y_test))

# R^2 changes on the data:
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr, y_train)
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
# plt.show()

# Function for interacting with data (NOT WORKING):
# def f(order, test_data):
#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
#     pr = PolynomialFeatures(degree=order)
#     x_train_pr = pr.fit_transform(x_train[['horsepower']])
#     x_test_pr = pr.fit_transform(x_test[['horsepower']])
#     poly = LinearRegression()
#     poly.fit(x_train_pr, y_train)
#     PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
#     plt.show()
#
#
# interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))
#
# pr1 = PolynomialFeatures(degree=2)
# x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# print(x_train_pr1.shape)
# poly1 = LinearRegression().fit(x_train_pr1, y_train)
# yhat_test1 = poly1.predict(x_test_pr1)
# Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
# # the model is not perfect on the $10,000 range and on the $30,000 to $40,000

# PART 3 - Ridge Regression:
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

# select the value of Alpha that minimizes the test error:
Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0, 1000))
for alpha in Alpha:
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))

width = 10
height = 6
plt.figure(figsize=(width, height))

plt.plot(Alpha, Rsqu_test, label='validation data  ')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
# plt.show()

RigeModel = Ridge(alpha=10)
RigeModel.fit(x_train_pr, y_train)
print('Ridge model score: ', RigeModel.score(x_test_pr, y_test))

# PART 4 - Grid Search:
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR = Grid1.best_estimator_
print('Best Grid score: ', BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))
