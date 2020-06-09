import numpy as np

np.random.seed(100)

dim = 6
iterations = 20
lr = 10**-2

#input data
X = np.random.rand(10,dim) #contains 10 training data points.
Y = np.append(np.ones((5,1)),np.zeros((5,1)), axis=0) #first 5 are labeled as 1, next 5 as 0.

#initialize parameters

def initialize_bias(X):
    
    X = np.append(X, np.ones((X.shape[0],1)), axis=1) 
    
    return X

def initialize_weights(dim):
    
    W = np.zeros((dim,1))
    
    return W
    
def sigmoid(z):
    
    activation = 1 / (1+ np.exp(-z))
    
    return activation


X = initialize_bias(X)
W = initialize_weights(X.shape[1])

def forward_propogation(X,Y,W):
    
    m = X.shape[0]
    matmul = np.dot(X,W)  # (10,7) X (7,1) = (10,1) - Same as Y.shape
    activation = sigmoid(matmul)# (10,1)
    
    error = Y*np.log(activation) + (1-Y)np.log(1-activation)
    
    loss = (-1 / m)*sum(error)
    
    gradient = (1/m)*np.dot(X.T, activation-Y) # (7,10) X (10X1) = (7,1) - Same as W.shape
    
    return loss, gradient
  

#the main loop
def gradient_descent(X,Y,W,lr,iterations):
    
    loss_ = []
    
    for i in range(iterations):
        
        loss, gradient = forward_propogation(X,Y,W)
        
        W = W - lr*gradient
        
        loss_.append(loss)
        
    return W, loss_
        