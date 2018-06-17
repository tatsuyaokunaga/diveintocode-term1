import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import*
import warnings
warnings.filterwarnings('ignore')


class LinearModel:
    def __init__(self, X, y, theta, iterations, alpha):

        self.X = X
        self.y = y
        self.theta = theta
        self.iterations = iterations
        self.alpha = alpha

    def compute_cost(self, X, y, theta):  # コスト関数
        m = len(self.y)
        h = np.dot(self.X, self.theta).flatten()
        J = 1 / (2 * m) * np.sum(np.square(h - self.y))
        return J

    def gradient_descent(self, X, y, theta, iteration, alpha):  # 最急降下法
        past_costs = []
        past_thetas = [self.theta]
        for i in range(self.iterations):
            m = len(X)
            h = np.dot(X, self.theta).flatten()
            self.theta -= (self.alpha / m) * (np.sum(np.dot(h - self.y, self.X)))
            past_costs.append(self.compute_cost(self.X, self.y, self.theta))
            past_thetas.append(self.theta)

        return past_costs  # , past_thetas

    def plot_learning_curve(self, X, y, theta, iteration, alpha):  # 学習曲線のプロット
        plt.figure(figsize=(12, 8))
        plt.title("Cost Function J")
        # Plot lines
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.plot(self.gradient_descent(self.X, self.y, self.theta, self.iterations, self.alpha))
        # Visualize


train_df = pd.read_csv('./train.csv')
train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)
train_df = train_df.drop(train_df[train_df['Id'] == 524].index)

x1 = np.array(train_df['GrLivArea']).reshape(-1, 1)
x2 = np.array(train_df['YearBuilt']).reshape(-1, 1)
y = train_df['SalePrice']
X = np.concatenate((x1, x2), axis=1)

min = X.min(axis=None, keepdims=True)
max = X.max(axis=None, keepdims=True)
X = (X - min) / (max - min)
X = np.insert(X, 0, 1, axis=1)
# yをarray変換
y = y.values
# y_numを正規化
min = y.min(axis=None, keepdims=True)
max = y.max(axis=None, keepdims=True)
iteration = 500
y = (y - min) / (max - min)
alpha = 0.01
np.random.seed(seed=10)
theta = np.random.rand(3, 1)
model = LinearModel(X, y, theta, iteration, alpha)  # (X,y,theta,500, 0.01
model.compute_cost(X, y, theta)  # X,y,theta
model.gradient_descent(X, y, theta, iteration, alpha)
model.plot_learning_curve(X, y, theta, iteration, alpha)