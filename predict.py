import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

url = "/Users/leandro998/Downloads/Data Analysis with Python - IBM/imports-85.data.csv"
df = pd.read_csv(url, header=None)
header = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
          "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length",
          "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
          "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df.columns = header
df = df.replace('?', np.NaN)
df = df.dropna(subset=["price"], axis=0)
# convert string into float:
df["price"] = df["price"].astype("float")
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df["horsepower"].replace(np.NaN, avg_horsepower, inplace=True)
df["horsepower"] = df["horsepower"].astype("float")
avg_bore = df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.NaN, avg_bore, inplace=True)
df["bore"] = df["bore"].astype("float")
avg_normalized_losses = df['normalized-losses'].astype('float').mean(axis=0)
df["normalized-losses"].replace(np.NaN, avg_normalized_losses, inplace=True)
df["normalized-losses"] = df["normalized-losses"].astype("float")
# print(df.dtypes)
# dtypes: 'horsepower': 'float64', 'curb-weight': 'int64', 'engine-size': 'int64', 'highway-mpg': 'int64'

# Multiple features regression:
lr = LinearRegression()
z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = df['price']
lr.fit(z, y)
yhat = lr.predict(z)
mse = mean_squared_error(df['price'], yhat)
print('The R^2 for multiple features regression is: ', lr.score(z, y), ' The MSE is: ', mse)
z1 = pd.DataFrame({"horsepower": [150], "curb-weight": [2284], 'engine-size': [200], 'highway-mpg': [22]})
yhat0 = lr.predict(z1)
print('The sale price using multiple regression is: ', yhat0, '\n')

# Check if a polynomial features can elaborate a higher R^2:
x_train, x_test, y_train, y_test = train_test_split(z, y, test_size=0.1, random_state=1)

Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    lr.fit(x_train_pr, y_train)
    Rsqu_test.append(lr.score(x_test_pr, y_test))

print('The result array indicates that 3 degree has a high R^2 value: ', Rsqu_test)

# 3 degree Polynomial Feature will have a better prediction. We must create new Linear Regression to compare the prices
pr = PolynomialFeatures(degree=3)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
# Create new Linear Regression to have 2 models to compare
lr2 = LinearRegression()
lr2.fit(x_train_pr, y_train)
yhat2 = lr2.predict(x_test_pr)
yhat3 = lr2.predict(x_train_pr)
mse2 = mean_squared_error(y_test, yhat2)
mse3 = mean_squared_error(y_train, yhat3)
# Print an array of prices predicted by the model:
# print('yhat2: ', yhat2, '\n yhat3: ', yhat3)
print('the R^2 for poly test is: ', lr2.score(x_test_pr, y_test), 'the MSE is: ', mse2)
print('the R^2 for poly train is: ', lr2.score(x_train_pr, y_train), 'the MSE is: ', mse3)

# To maximize the R^2 and minimize the MSE we must use Polynomial Features instead of multiple features

# FIX:

# To predict the price for a car we must input the values of the car we will sell:
x_input = pd.DataFrame({"horsepower": [150], "curb-weight": [2284], 'engine-size': [200], 'highway-mpg': [22]})
x_tr = pr.fit_transform(x_input)
yhat4 = lr2.predict(x_tr)
print('the sale price using polynomial features should be: ', yhat4)
