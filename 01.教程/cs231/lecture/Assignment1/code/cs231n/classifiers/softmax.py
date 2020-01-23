import numpy as np
from random import shuffle
import math


def softmax_loss_naive(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores) #prevents numerical instability
        correct_class_score = scores[y[i]]

        exp_sum = np.sum(np.exp(scores))
        loss += np.log(exp_sum) - correct_class_score

        dW[:,y[i]] += -X[i,:]
        for j in range(num_classes):
            dW[:,j] += X[i,:] * (np.exp(correct_class_score) / np.sum(np.exp(scores)))


    loss /= num_train
    loss += 0.5 * reg * np.sum( W*W )
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    scores = np.dot(X,W)        # m*10
    scores = (scores.T - np.max(scores, axis=1)).T      # m*10
    correct_class_score = scores[range(num_train), y]   # m*1

    exp_sum = np.sum(np.exp(scores), axis=1)            # (m,)
    loss = np.sum(np.log(exp_sum) - correct_class_score)

    # Sample * Class: m*10
    coMatrix = np.zeros((num_train,num_classes))
    coMatrix[range(num_train),y] = -1
    coMatrix = (coMatrix.T + np.exp(correct_class_score) / exp_sum).T
    dW = np.dot(X.T, coMatrix)

    loss /= num_train
    loss += np.sum(reg * W*W)
    dW /= num_train
    dW += reg * W
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
