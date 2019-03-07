import numpy as np
class LinearRegression():
    def __init__(self,diff=100):
        self.diff = diff
        
        
    def h(self,x,theta):
        return np.dot(x,theta)


    def cost_fuction(self,x,y,theta,m):
        loss = self.h(x,theta) - y
        error = (1/(2 * m)) * np.sum(np.square(loss))
        return error


    def gradient(self,x,y,theta,m):
        loss = self.h(x,theta) - y
        grad = (1/m)*np.dot(x.T,loss)
        return grad


    def mini_batch_gradient_descent(self,x,y,theta,learning_rate,itration,batch_size,m):
        print("Start error :",self.cost_fuction(x,y,theta,m))
        for i in range(itration):
            for j in range(0,m,batch_size):
                x_batch = x[j:j+batch_size,:]
                y_batch = y[j:j+batch_size,:]
                grad = self.gradient(x_batch,y_batch,theta,m)
                theta = theta - learning_rate * grad
            if i % self.diff == 0 :
                print("itration : ",i," error : ",self.cost_fuction(x,y,theta,m))
        print("End error :",self.cost_fuction(x,y,theta,m))
        return theta



    def fit(self,x,y,learning_rate=0.001,itration=1000,batch_size=20):
        print("learning_rate : ",learning_rate , " itration : ",itration, " batch_size : ",batch_size)
        m,n = x.shape
        ones = np.ones([m,1])
        x = np.concatenate((ones,x),axis=1)
        theta = np.zeros([n+1,1])
        theta = self.mini_batch_gradient_descent(x,y,theta,learning_rate,itration,batch_size,m)
        return theta


    def predict(self,x,theta):
        m,n = x.shape
        ones = np.ones([m,1],dtype=np.int32)
        x = np.concatenate((ones,x),axis=1)
        y_pred = np.dot(x,theta)
        return y_pred