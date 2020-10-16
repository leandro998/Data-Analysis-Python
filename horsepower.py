import pandas as pd
import numpy as np
from scipy import stats
url = "/Users/leandro998/Downloads/Data Analysis with Python - IBM/imports-85.data.csv"
df = pd.read_csv(url, header=None)
header = ["symboling","normalized-losses","make","fuel-type","aspiration",
          "num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length",
          "width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system",
          "bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

df.columns = header
df = df.replace('?', np.NaN)
df = df.dropna(subset=["price"], axis=0)
df["price"] = df["price"].astype("float")

# we must create a variable to store the mean and latter replace the original column to the mean variable with "inplace=True"
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
df['horsepower'].replace(np.NaN, avg_horsepower, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

