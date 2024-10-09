
import pandas as pd
import numpy as np

def Synthetic_Data(n):
    x = np.random.rand(n, 1)
    y = 2+3*x + (np.random.rand(n, 1)/5)
    return pd.DataFrame(np.hstack([x ,y]), columns = ['x', 'y'])
df = Synthetic_Data(1000)

x = df.iloc[:, :-1]
y = df.iloc[:, 1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state = 0)
iterations = 1000
learning_rate = 0.01

def predict(weights, bias, x):
    return x.dot(weights)+bias

def update_weights(weights, bias, x, y, m):
    y_pred = predict(weights, bias, x)
    dw = -(2*(x.T).dot(y - y_pred))/m
    db = -2*np.sum(y - y_pred)/m
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    return weights, bias

def fit(x, y):
    m, n = x.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(iterations):
        weights, bias = update_weights(weights, bias, x, y, m)
    return weights, bias

weights, bias = fit(x_train, y_train)

y_pred = predict(weights, bias, x_test)

from sklearn.metrics import r2_score, mean_squared_error

print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color = 'orange')
plt.show()
