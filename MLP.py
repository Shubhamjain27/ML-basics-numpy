

h = 100 # size of hidden layer
#Shape of weight matrix is (input-dimesnions/features, no. of output classes)
W1 = 0.01 * np.random.randn(n_dims, h) 
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, n_classes)
b2 = np.zeros((1, n_classes))


#one hidden layer followed by the output layer.

def relu(z):
    return np.maximum(0,z)

def softmax(z):
    
    #if we do not pass the axis parameter, the default is axis=None, which means it will take sum
    #over the whole matrix, which we do not want.
    
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def softmax_stabalized(z):
    
    #we subtract the max value from each value in the matrix to avoid overflow - e^x grows pretty fast.
    
    e_x = np.exp(z-np.max(z))
    return e_x/np.sum(e_x, axis=1, keepdims=True)

num_examples = X.shape[0]
for i in range(10000):
    
    matmul_hidden = np.dot(X,W1) + b1 # (n, n_dims) X (n_dims, h) = (n,h)
    hidden_activation = relu(matmul) #relu activation
    
    matmul_output = np.dot(hidden_activations,W2) + b2 # (n,h) X (h, n_classes) = (n, n_classes)
    
    probs = softmax_stabalized(matmul_output) # (n, n_classes)
    
    
    #y_train is not one-hot encoded. It contains the class lables as numerically encoded. eg - [1,1,3,3,2,2]
    #We use multidimensional array indexing to extract softmax probability of the correct label for each sample.
    #We basically use values in y_train as an index in the probs matrix(n, n_classes). For each row in the probs matrix,
    #which correspond to each sample, the y_train value represents the index of the correct class. The value at this 
    #index in the row is the predicted probability of the correct class. 
    #As per the formula, for each sample, we need to take sum over log(all predicted probabilities) and true probability
    #corresponding to the predicted probability(which is either 1 or 0). 
    #So our task is to just find the correct predicted probability (true probability for it will be 1), take it's log,
    #and then sum for all samples. That gives our loss. 
    
    #https://numpy.org/doc/stable/user/basics.indexing.html#indexing-multi-dimensional-arrays

    corect_logprobs = -np.log(probs[range(num_examples), y_train])
    data_loss = np.sum(corect_logprobs) / num_examples
    
    
    # compute the gradient on scores
    #the backprop used is explained here
    #https://www.notion.so/shubhamjain/Backprop-Implementation-630c0fccf2014aa6a0f745f8e37c9aed
    
    dscores = probs
    dscores[range(num_examples), y_train] -= 1
    dscores /= num_examples
    
    dW2 = np.dot(hidden_activation.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    
    dhidden = np.dot(dscores,W2.T)

    #backprop into relu activation - making less than zero values=0, other values remain same.
    
    dhidden[hidden_activation <=0] = 0
    
    dW2 = np.dot(X.T, dhidden)
    db2 = np.sum(dhidden, axis=0, keepdims=True)
    
    #gradient descent
    
    W1 -= alpha*dW1
    b1 -= alpha*db1
    W2 -= alpha*dW2
    b2 -= alpha*db2