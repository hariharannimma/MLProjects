import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

data = pd.read_excel('archive/CleanedData.xlsx')

X, y = data['Open'].values, data['Close'].values
X = X.reshape(-1, 1)
X = np.array(X, dtype='float64')
y = np.array(y, dtype='float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print(f"Shape of X: {X.shape} \n and shape of y: {y.shape}")
print(f"Shape of X_train: {X_train.shape}, \n Shape of X_test: {X_test.shape}, \n Shape of y_train: {y_train.shape}, \n Shape of y_test: {y_test.shape}")

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

modal = LinearRegression(lr = 0.09, n_iter=100)
modal.fit(X_train, y_train)
predict = modal.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test ,predict)
