import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

# Step 1 - convert drive-wheels to index:
dummy_variable_1 = pd.get_dummies(df["drive-wheels"])
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("drive-wheels", axis=1, inplace=True)
print(df.head())

# Step 2 - create model:
x_data = df.drop('price', axis=1)
y_data = df['price']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

lre1 = LinearRegression()
lre1.fit(x_train[['horsepower', 'fwd', 'curb-weight']], y_train)
R2 = lre1.score(x_train[['horsepower', 'fwd', 'curb-weight']], y_train)
print('The R^2 is: ', R2)
X = pd.DataFrame({"horsepower": [150], "fwd": [1], 'curb-weight': [2284]})
Yhat = lre1.predict(X)
print('The predicted price for the car should be: ', Yhat)