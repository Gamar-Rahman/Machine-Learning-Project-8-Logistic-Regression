import numpy as np

np.random.seed(1)


#Problem 1.2
def sigmoid(z):
    """
    sigmoid function that maps inputs into the interval [0,1]
    Your implementation must be able to handle the case when z is a vector (see unit test)
    Inputs:
    - z: a scalar (real number) or a vector
    Outputs:
    - trans_z: the same shape as z, with sigmoid applied to each element of z
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    trans_z = 1 / (1 + np.exp(-z))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return trans_z

def logistic_regression(X, w):
    """
    logistic regression model that outputs probabilities of positive examples
    Inputs:
    - X: an array of shape (num_sample, num_features)
    - w: an array of shape (num_features,)
    Outputs:
    - logits: a vector of shape (num_samples,)
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    logits = sigmoid(X @ w)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return logits

#Problem 1.3
def logistic_loss(X, w, y):
    """
    a function that compute the loss value for the given dataset (X, y) and parameter w;
    It also returns the gradient of loss function w.r.t w
    Here (X, y) can be a set of examples, not just one example.
    Inputs:
    - X: an array of shape (num_sample, num_features)
    - w: an array of shape (num_features,)
    - y: an array of shape (num_sample,), it is the ground truth label of data X
    Output:
    - loss: a scalar which is the value of loss function for the given data and parameters
    - grad: an array of shape (num_featues,), the gradient of loss
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pred = logistic_regression(X, w)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    grad = X.T @ (pred - y) / X.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, grad

#Problem 2.5
def train_model_gd():
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    np.random.seed(1)

    # load the digits dataset
    digits = load_digits(n_class=2)
    ones = np.ones(digits.data.shape[0]).reshape(-1, 1)
    digits.data = np.concatenate((ones, digits.data), axis=1)
    from sklearn.preprocessing import StandardScaler

    

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8, random_state=1)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    num_iters = 200
    lr = 0.1

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    w = np.random.rand(X_train.shape[1])

    for _ in range(num_iters):
        loss, grad = logistic_loss(X_train, w, y_train)
        w = w - lr * grad

    y_pred = (logistic_regression(X_test, w) > 0.5).astype(int)
    acc = np.mean(y_pred == y_test)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc

#Problem 3
def softmax(x):
    """
    Convert logits for each possible outcomes to probability values.
    In this function, we assume the input x is a 2D matrix of shape (num_sample, num_classes).
    So we need to normalize each row by applying the softmax function.
    Inputs:
    - x: an array of shape (num_sample, num_classse) which contains the logits for each input
    Outputs:
    - probability: an array of shape (num_sample, num_classes) which contains the
                    probability values of each class for each input
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    probability = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return probability

def MLR(X, W):
    """
    performs logistic regression on given inputs X
    Inputs:
    - X: an array of shape (num_sample, num_feature)
    - W: an array of shape (num_feature, num_class)
    Outputs:
    - probability: an array of shape (num_sample, num_classes)
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    probability = softmax(X @ W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return probability

#Problem 3.1
def cross_entropy_loss(X, W, y):
    """
    Inputs:
    - X: an array of shape (num_sample, num_feature)
    - W: an array of shape (num_feature, num_class)
    - y: an array of shape (num_sample,)
    Ouputs:
    - loss: a scalar which is the value of loss function for the given data and parameters
    - grad: an array of shape (num_featues, num_class), the gradient of the loss function
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    probability = MLR(X, W)
    probability = np.clip(probability, 1e-15, 1 - 1e-15)

    n = X.shape[0]
    Y = np.zeros((n, W.shape[1]))
    Y[np.arange(n), y] = 1

    loss = -np.sum(Y * np.log(probability)) / n
    grad = X.T @ (probability - Y) / n
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, grad

#Problem 3.2
def learn_real_dataset():
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(1)

    digits = load_digits()
    ones = np.ones(digits.data.shape[0]).reshape(-1, 1)
    digits.data = np.concatenate((ones, digits.data), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, train_size=0.8, random_state=1
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    num_iters = 200
    lr = 0.1
    num_classes = len(np.unique(y_train))

    W = np.random.rand(X_train.shape[1], num_classes)

    for _ in range(num_iters):
        loss, grad = cross_entropy_loss(X_train, W, y_train)
        W = W - lr * grad

    y_pred = np.argmax(MLR(X_test, W), axis=1)
    acc = np.mean(y_pred == y_test)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc