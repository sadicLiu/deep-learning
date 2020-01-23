from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
  
    In other words, the network has the following architecture:
  
    input - fully connected layer - ReLU - fully connected layer - softmax
  
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
    
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
    
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        # self.params['W1'] = std * np.random.randn(input_size, hidden_size) / np.sqrt(2.0/(input_size))
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        # self.params['W2'] = std * np.random.randn(hidden_size, output_size) / np.sqrt(2.0/(hidden_size))
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
    
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
    
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
    
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        # ReLu激活函数只在hidden layer使用
        hidden = np.maximum(0, np.dot(X, W1) + b1) * 1.0
        scores = np.dot(hidden, W2) + b2
        pass

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None

        # data loss
        # 减去最大值会使下面的计算更稳定，且对结果没有影响
        scores_max = np.max(scores,axis=1,keepdims=True) * 1.0
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, dtype=float, axis=1, keepdims=True)
        # print(np.sum(exp_scores, axis=1, keepdims=True))
        corect_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(corect_logprobs) / N

        loss = data_loss + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)

        pass

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N
        dW2 = np.dot(hidden.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden <= 0] = 0
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        dW2 += reg * W2
        dW1 += reg * W1

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        pass

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        # 这个参数的意思是所有样本完成一次正向传递和反向传递需要多少次迭代
        # 如果num_train < batch_size(toy data),这个参数设为1
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        # beta1 = 0.9
        # beta2 = 0.999
        # eps = 1e-8
        # mW1 = np.zeros_like(self.params['W1'])
        # mW2 = np.zeros_like(self.params['W2'])
        # mb1 = np.zeros_like(self.params['b1'])
        # mb2 = np.zeros_like(self.params['b2'])
        # vW1 = np.zeros_like(self.params['W1'])
        # vW2 = np.zeros_like(self.params['W2'])
        # vb1 = np.zeros_like(self.params['b1'])
        # vb2 = np.zeros_like(self.params['b2'])
        #
        # mu_all = [0.5,0.9,0.95,0.99]
        # mu = mu_all[0]

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx,:]
            y_batch = y[idx]

            pass

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            dW1 = grads['W1']
            dW2 = grads['W2']
            db1 = grads['b1']
            db2 = grads['b2']
            self.params['W1'] -= learning_rate*dW1
            self.params['W2'] -= learning_rate*dW2
            self.params['b1'] -= np.reshape(learning_rate*db1, self.params['b1'].shape)
            self.params['b2'] -= np.reshape(learning_rate*db2, self.params['b2'].shape)

            # 梯度更新，这里使用NN_3中介绍的一种方法：Adam
            # vW1 = mu * vW1 -learning_rate * dW1
            # vW2 = mu * vW2 - learning_rate * dW2
            # vb1 = mu * vb1 - learning_rate * db1
            # vb2 = mu * vb2 - learning_rate * db2
            # self.params['W1'] += vW1
            # self.params['W2'] += vW2
            # self.params['b1'] += np.reshape(vb1, self.params['b1'].shape)
            # self.params['b2'] += np.reshape(vb2, self.params['b2'].shape)

            # mW1 = beta1*mW1 + (1-beta1)*dW1
            # vW1 = beta2*vW1 + (1-beta2)*(dW1**2)
            # self.params['W1'] += -learning_rate * mW1 / (np.sqrt(vW1) + eps)
            # mW2 = beta1*mW2 + (1-beta1)*dW2
            # vW2 = beta2*vW2 + (1-beta2)*(dW2**2)
            # self.params['W2'] += -learning_rate * mW2 / (np.sqrt(vW2) + eps)
            # mb1 = beta1*mb1 + (1-beta1)*db1
            # vb1 = beta2*vb1 + (1-beta2)*(db1**2)
            # self.params['b1'] += np.reshape(-learning_rate * mb1 / (np.sqrt(vb1) + eps), (self.params['b1'].shape))
            # mb2 = beta1*mb2 + (1-beta1)*db2
            # vb2 = beta2*vb2 + (1-beta2)*(db2**2)
            # self.params['b2'] += np.reshape(-learning_rate * mb2 / (np.sqrt(vb2) + eps), (self.params['b2'].shape))

            pass

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

            # if it>50 and it<=100 :
            #     mu = mu_all[1]
            # elif it>100 and it<=150:
            #     mu = mu_all[2]
            # elif it>150:
            #     mu = mu_all[3]

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }


    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
    
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        hidden = np.dot(X,self.params['W1']) + self.params['b1']
        scores = np.dot(hidden,self.params['W2']) +self.params['b2']
        # maxScores = np.max(scores, axis=1, keepdims=True)
        # [n,c] = np.where(scores == maxScores)
        y_pred = np.argmax(scores,axis=1)

        pass
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
