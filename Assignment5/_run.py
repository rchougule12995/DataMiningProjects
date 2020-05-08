import pandas as pd
import matplotlib.pyplot as mp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./specs/markB_question.csv')

print(df.head())

X = df.iloc[:, 0].values.reshape(-1, 1)
Y = df.iloc[:, 2].values.reshape(-1, 1)

# print(X)
linear_reg = LinearRegression()
linear_reg.fit(X, Y)
Y_pred = linear_reg.predict(X)

mp.scatter(X, Y)
mp.plot(X, Y_pred, color='red')

#for index, row in df.iterrows():
df['final_linear'] = linear_reg.predict(X)

print("***********************************************")
print(df['final_linear'])
print("***********************************************")
print(linear_reg.score(X, Y))
print("***********************************************")
print(linear_reg.coef_)
print("***********************************************")
print(linear_reg.intercept_)
print("***********************************************")


poly = PolynomialFeatures(2)

test = poly.fit_transform(X)

print(test)


#mp.show()

