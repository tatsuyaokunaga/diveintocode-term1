import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import*
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class LinearModel():

    def __init__(self,X,y,theta,ramda=0.05,iterations=1000,alpha = 0.025,threshold=0.5):
        np.random.seed(seed=10)
        theta=np.random.rand(5,1)
        self.X = X
        self.y = y
        self.theta=theta
        self.iterations=iterations
        self.alpha=alpha
        self.threshold=threshold
        self.ramda = ramda
    
    def sigmoid(self):
        theta_x = np.dot(self.X,self.theta)
        return 1/(1+np.exp(-self.theta_x))

    def compute_cost(self):
       
        h = sigmoid(self.X,self.theta)
        m = len(self.X)
        self.theta[0]=0
        J = (1/m)*(np.dot(-self.y.T,np.log(h)) - np.dot((1-self.y).T,np.log(1-h)))+\
                                   (ramda/2*m)*(self.theta**2).sum()
        return J[0,0]

    def gradient_descent(self):
        m = len(self.X)
        past_costs =[]
        for i in range(self.iterations):
            past_costs.append(compute_cost(self.X, self.y, self.theta))
            h_x_y = sigmoid(self.X , self.theta) - self.y
            theta  = self.theta - (self.alpha/m) * (np_dot + self.ramda*self.theta)
        
        return past_costs ,theta

    def plot_learning_curve(self):
        plt.figure(figsize=(12,8))
        plt.title("Cost Function J")
        # Plot lines
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.plot(self.past_costs) 
   
    # 確率を求める
    def predict_probs(self):
        return sigmoid(self.X,self.theta)
        
    # 分類を行う。
    def predict(self):
        """
        threshold: 閾値
        """""
        theta_x = np.dot(self.X,self.theta)
        pred = np.sum((sigmoid(self.X,self.theta)>=self.threshold)[:50])  
    
        return pred
    
        
train_df = pd.read_csv("iris/Iris.csv")
y = train_df[["Species"]]
y=y.reset_index(drop=True)
y=pd.get_dummies(y["Species"])
y = y.astype('int8')
y = y.reset_index(drop=True)
y = y.drop("Iris-virginica",axis=1)
X = X.iloc[50:,:]
X = X.reset_index(drop=True)

model=LinearModel(X,y,theta)