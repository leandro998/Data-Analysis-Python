# This file contains the code for labs 1 to 4 and general tests while listening to the videos
import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
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

# the native url informed is not working because it downloads a file, instead of open it on the browser:
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
url = "/Users/leandro998/Downloads/Data Analysis with Python - IBM/imports-85.data.csv"

# create a variable df (dataframe) to read the csv using pandas and not storing the first line as header:
df = pd.read_csv(url, header=None)

# the headers are available in the 2 file:
header = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
          "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length",
          "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
          "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df.columns = header
# print the results of the 5 top elements:
# print(df.head())
# pd.set_option('display.width', 320) # extends the standard size of 80
# pd.set_option('display.max_columns', 10) # extends the standard number of columns

# code to store the information to a new csv file in the given path:
# path = "/Users/leandro998/Downloads/Data Analysis with Python - IBM/car_database.data.csv"
# df.to_csv(path)

# show the data type (object, int, float, etc):
# print(df.dtypes)

# describe the summary stats of the database for a quick view:
# print(df.describe())
# print("describe all")
# print(df.describe(include="all"))
# print(df.info)

# to remove NaN (axis=0 > drop row, axis=1 > drop column):
# df.dropna(subset=["price"], axis=0, inplace=True), but you have to make sure that ? or " " are registered as NaN
# change ? or blanks to NaN:
df = df.replace('?', np.NaN)
# drop the rows where column price = NaN:
df = df.dropna(subset=["price"], axis=0)
# df.dropna(subset=["price"], axis=0, inplace=True) >> where inplace changes the current list
# check the count:
# print(df.count)

# check if a value isNull:
# missing_data = df.isnull()
# print(missing_data.head(5))
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print (missing_data[column].value_counts())
#     print("")

# to see how many values are in a column: >> insert .idxmax() at the end to show the max count id:
# print(df['num-of-doors'].value_counts())

# convert mpg to L/km:
# df["city-mpg"] = 235/df["city-mpg"]
# df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)
# print(df["city-L/100km"])
# print(df.head())

# convert data type (first we must set "?" or " " to a valid number, and latter convert):
# df = df.replace('?', 0)
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

# Group By Question:
# df_group_one = df[['drive-wheels', 'price', 'horsepower', 'highway-mpg']]
# df_group_one['horsepower'] = df_group_one["horsepower"].astype("float")
# print(df_group_one.dtypes)
# df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
# print(df_group_one)

# print(df.dtypes)

# quiz:
# df["price"] = df["price"]+1
# print(df["price"].head())

# binning: define a range to categorize a set of values
# 3 steps prior: convert blank to NaN + drop rows NaN + convert to float >> range from 5.118, 18.545, 31.972, 45.400
# bins = np.linspace(min(df["price"]), max(df["price"]), 4)
# group_names = ["Low", "Medium", "High"]
# print(bins)
# df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)
# print(df[["price", "price-binned"]])
# Question: how to plot this in a distribution chart?
# pyplot.bar(group_names, df["price-binned"].value_counts())
# plt.pyplot.xlabel("price")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("price bins")
# plt.pyplot.show()

# horsepower lab:
# df["horsepower"] = df["horsepower"].astype(float)
# %matplotlib inline
# plt.pyplot.hist(df['horsepower'])
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
# plt.pyplot.show()

# Transform string in reference integer:
# test = pd.get_dummies(df["body-style"])
# print(test)

# value counts:
# drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
# drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# print(drive_wheels_counts)
# sns.boxplot(x='drive-wheels', y='price', data=df) #>> what does sns means?

# Scatterplot:
# y = df['price']
# x = df['engine-size']
# plt.pyplot.scatter(x, y)
# plt.pyplot.xlabel("engine-size")
# plt.pyplot.ylabel("price")
# plt.pyplot.title("price per engine size")
# plt.pyplot.show()

# Scatterplot 2 (seaborn >> with the tendency line)
# sns.regplot(x='engine-size', y='price', data=df)
# sns.regplot(x='highway-mpg', y='price', data=df)
# plt.pyplot.ylim(0,)
# plt.pyplot.show()
# Residual plot:
# sns.residplot(df['highway-mpg'], df['price'])
# plt.pyplot.ylim(0,)
# plt.pyplot.show()
# Distribution plot:
# axl = sns.distplot(df['price'], hist=False, color='r', label= 'actual value')
# sns.distplot(Yhat, hist=False, color='b', label='fitted value', ax=axl)


# Group By:
# df_test = df[['drive-wheels', 'body-style', 'price']]
# df_group = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
# df_pivot = df_group.pivot(index='drive-wheels', columns='body-style')
# # replace NaN with 0
# df_pivot = df_pivot.fillna(0)
# # print(df_pivot)
# # # Heatmap:
# # unnamed labels >>
# # plt.pyplot.pcolor(df_pivot, cmap="RdBu")
# # plt.pyplot.colorbar()
# # plt.pyplot.xlabel("body-style")
# # plt.pyplot.ylabel("drive-wheels")
# # named labels >>
# fig, ax = plt.pyplot.subplots()
# im = ax.pcolor(df_pivot, cmap='RdBu')
# # label names
# row_labels = df_pivot.columns.levels[1]
# col_labels = df_pivot.index
# # move ticks and labels to the center
# ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
# # insert labels
# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(col_labels, minor=False)
# # rotate label if too long
# plt.pyplot.xticks(rotation=90)
# fig.colorbar(im)
# plt.pyplot.show()

# Correlation and P-Value:
# pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
# pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )
# pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
# print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
# pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value)
# pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# pearson_coef, p_value = stats.pearsonr(df['normalized-losses'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# ANOVA:
# df_gptest = df[['drive-wheels','body-style','price']]
# grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
# print(grouped_test1.head())
# grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
# print(grouped_test2.get_group('4wd')['price'])
# when analizing the f_val and p_val from distincts groups the results may change:
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],
#                               grouped_test2.get_group('rwd')['price'],
#                               grouped_test2.get_group('4wd')['price'])
# print("ANOVA results: F=", f_val, ", P =", p_val)

# Linear Regression
# lm = LinearRegression()
# x = df[['highway-mpg']]
# y = df[['price']]
# lm.fit(x, y)
# Yhat = lm.predict(x)
# MSE = Mean Squared Error
# print('the output of the first 5 values are: ' + str(Yhat[0:5]))
# mse = mean_squared_error(df['price'], Yhat)
# print('The mean square error of price and predicted value is: ', mse)
# print('intercept: ' + str(lm.intercept_) + ' coef: ' + str(lm.coef_))
# print('R-Square is: ' + str(lm.score(x, y)))
# lml = LinearRegression()
# lml.fit(df[['engine-size']], df[['price']])
# print('engine size: intercept: ' + str(lml.intercept_) + ' coef: ' + str(lml.coef_))


# Multiple Regression
# lm2 = LinearRegression()
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# lm2.fit(Z, df['price'])
# print('coef' + str(lm2.coef_) + ' intercept: ' + str(lm2.intercept_))
# lm2.fit(df[['normalized-losses', 'highway-mpg']], df['price'])
# print('coef' + str(lm2.coef_) + ' intercept: ' + str(lm2.intercept_))
# predicting values:
# Y_hat = lm2.predict(Z)
# width = 6
# height = 5
# plt.pyplot.figure(figsize=(width, height))
# # red = actual values / blue = fitted values:
# ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
# plt.pyplot.title('Actual vs Fitted Values for Price')
# plt.pyplot.xlabel('Price (in dollars)')
# plt.pyplot.ylabel('Proportion of Cars')
# plt.pyplot.show()

# Polynomial Model:
# def PlotPolly(model, independent_variable, dependent_variabble, Name):
#     x_new = np.linspace(15, 55, 100)
#     y_new = model(x_new)
#
#     plt.pyplot.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
#     plt.pyplot.title('Polynomial Fit with Matplotlib for Price ~ Length')
#     ax = plt.pyplot.gca()
#     ax.set_facecolor((0.898, 0.898, 0.898))
#     fig = plt.pyplot.gcf()
#     plt.pyplot.xlabel(Name)
#     plt.pyplot.ylabel('Price of Cars')
#     plt.pyplot.show()
#
#
# x = df['highway-mpg']
# y = df['price']
# f = np.polyfit(x, y, 3)
# p = np.poly1d(f)
# print(p)
# print(np.polyfit(x, y, 3))
# PlotPolly(p, x, y, 'highway-mpg')
# r_squared = r2_score(y, p(x))
# print('The R-square value is: ', r_squared)
# print(mean_squared_error(df['price'], p(x)))

# Multiple Features: // different number of features after transformation:
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# y = df['price']
# pr = PolynomialFeatures(degree=2)
# Z_pr = pr.fit_transform(Z)
# print(Z.shape)
# print(Z_pr.shape)
# # Pipeline:
# Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
# pipe = Pipeline(Input)
# print(pipe)
# pipe.fit(Z, y)
# ypipe = pipe.predict(Z)
# print(ypipe[0:4])

# Prediction and Decision Making:
# new_input = np.arange(1, 100, 1).reshape(-1, 1)
# lm = LinearRegression()
# x = df[['highway-mpg']]
# y = df[['price']]
# lm.fit(x, y)
# yhat = lm.predict(new_input)
# print(yhat[0:5])
# plt.pyplot.plot(new_input, yhat)
# plt.pyplot.show()

# Train test model:
# x_data = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# y_data = df['price']
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# Cross-validation: splits the dataset into K groups and test with all of them.
# After training we compare the results and use the average:
# lr = LinearRegression()
# scores = cross_val_score(lr, x_data, y_data, cv=3)
# print(np.mean(scores))
# lr2e = LinearRegression()
# yhat = cross_val_predict(lr2e, x_data, y_data, cv=3)
# print(yhat)

# R-squared testing:
# Rsqu_test = []
# order = [1, 2, 3, 4]
# for n in order:
#     pr = PolynomialFeatures(degree=n)
#     x_train_pr = pr.fit_transform(x_train[['horsepower']])
#     x_test_pr = pr.fit_transform(x_test[['horsepower']])
#     lr.fit(x_train_pr, y_train)
#     Rsqu_test.append(lr.score(x_test_pr, y_test))
#
# print(Rsqu_test)

# Ridge Regression:
# RidgeModel = Ridge(alpha=0.1)
# x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# y = df['price']
# RidgeModel.fit(x, y)
# Yhat = RidgeModel.predict(x)
# print(Yhat)

# Grid Search CV:
