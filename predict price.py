import pandas
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

data = pandas.read_csv('Book 11.csv')
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[14]]))
print(model.predict([[18]]))
# plt.scatter(data['version'], data['price'])
# plt.show()
# print(data.head())
