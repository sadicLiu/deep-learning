import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    D:特征数（3073），C:类别数（10）
    N:样本数
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)  # 1*10
        correct_class_score = scores[y[i]]

        # 所有得分的margin
        margins = scores - correct_class_score + 1  # note delta = 1
        margins[y[i]] = 0

        # 求出当前样本的margin有几个大于0（为loss作了贡献）
        count = 0
        for m in range(num_classes):
            if margins[m] > 0:
                count += 1
        # 如果count>0，在这里就可以得出dW[:,y[i]]
        if count > 0:
            dW[:, y[i]] += -count * X[i, :].T

        # dW（梯度）是 3073*10 的矩阵
        # 下面求损失函数的同时，顺带求梯度
        for j in range(num_classes):
            if j == y[i]:
                continue
            if margins[j] > 0:
                loss += margins[j]
                dW[:, j] += X[i, :].T

    loss /= num_train
    loss += reg * np.sum(W * W)

    # 梯度dW是3073*10的矩阵
    # 梯度求平均
    dW /= num_train
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
  
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_classes = W.shape[1]
    num_train = X.shape[0]

    # 首先计算所有样本的得分
    scores = X.dot(W)  # m*10

    # 然后计算margins   # m*10
    idx = np.array(range(num_train))
    margins = (scores.T - scores[idx, y]).T + 1  # delta = 1
    margins[idx, y] = 0

    # margins中所有大于0的元素相加就是损失函数
    mask = margins > 0
    loss = np.sum(margins[mask])

    loss /= num_train
    loss += reg * np.sum(W * W)

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # count：magins中每行margin大于0的个数
    count = np.sum(mask == True, axis=1)    # (500,)
    np.reshape(count,(num_train,1))         # 把count转成m*1的矩阵
    margins[margins <= 0] = 0.0
    margins[margins > 0] = 1.0
    margins[idx,y] = -count

    dW = np.dot(X.T,margins)

    # 平均、正则化
    dW /= num_train
    dW += 2 * reg * W

    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW
