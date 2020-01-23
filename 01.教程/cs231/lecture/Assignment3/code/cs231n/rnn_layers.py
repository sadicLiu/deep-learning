from __future__ import print_function, division
from builtins import range
import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    cache = x, Wx, Wh, b, next_h, prev_h

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, Wx, Wh, b, next_h, prev_h = cache

    # 令next_h=tanh(z)
    dz = dnext_h * (1 - next_h ** 2)
    dx = np.dot(dz, Wx.T)
    dprev_h = np.dot(dz, Wh.T)
    dWx = np.dot(x.T, dz)
    dWh = np.dot(dz.T, prev_h).T
    db = np.sum(dz, axis=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N,T,H))
    cache = {}

    # 这个h0的处理要注意一下
    prev_h = h0
    for t in range(T):
        xt = np.reshape(x[:,t,:], (N,D))
        h[:,t,:], cache[t] = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        prev_h = h[:,t,:]

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    N, D = cache[0][0].shape
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))

    dh_next = np.zeros((N, H))
    for i in reversed(np.arange(T)):
        # 这里的dh很是令人费解，实际上，传入的dh是 dloss/dh
        # 如果我的理解没错的话，是指每一个state的loss对这个state的梯度
        # predict = V*state, 所有state共享参数V，而且h是所有state的矩阵，一次就能计算所有loss
        dh_cur = dh_next + dh[:, i, :]
        dx[:, i, :], dh_prev, tdWx, tdWh, tdb = rnn_step_backward(dh_cur, cache[i])
        dWx += tdWx
        dWh += tdWh
        db += tdb

        dh_next = dh_prev

    dh0 = dh_prev

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    # x中存的是样本中所有单词对应的索引
    # W中存的是样本中所有出现的单词对应的向量
    # 这个函数的作用就是将x从索引的形式转成向量的形式

    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    minibatches size N
    sequence length T
    vocabulary of V words
    dimension D

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # x是N个长度为T的序列的索引（整数）
    # W是将V个单词转成向量形式，每一行D维向量代表一个单词
    N, T = x.shape
    V, D = W.shape
    out = np.zeros((N, T, D))
    out[range(N),:,:] = W[x[range(N),:],:]
    cache = x, N, T, V, D

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################

    # 如果将x转成one-hot形式，则x（N, T, V）
    # out = x * W
    # dW = dout * x
    x, N, T, V, D = cache
    dim = N*T

    dout_reshape = np.reshape(dout, (dim, D))

    # 将x转成one-hot形式
    x_new  = np.zeros((dim, V))
    for i in range(dim):
        n = int(i / T)
        t = int(i % T)
        x_new[i, x[n,t]] = 1

    dW = np.dot(x_new.T, dout_reshape)

    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, D = x.shape
    _, H = prev_h.shape
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b # (N,4H)
    inputs = sigmoid(a[:,0:H])
    forgets = sigmoid(a[:,H:2*H])
    outputs = sigmoid(a[:,2*H:3*H])
    gate = np.tanh(a[:,3*H:])

    next_c = forgets*prev_c + inputs*gate
    next_h = outputs * np.tanh(next_c)

    cache = next_c, prev_c, prev_h, gate, inputs, outputs, forgets, N, D, H, Wx, Wh, b, x

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    next_c, prev_c, prev_h, gate, inputs, outputs, forgets, N, D, H, Wx, Wh, b, x = cache

    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros((4*H,))

    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * outputs * (1-np.tanh(next_c)**2)
    df = dnext_c * prev_c
    di = dnext_c * gate
    dg = dnext_c * inputs

    dprev_c = dnext_c * forgets
    da1 = di * inputs * (1-inputs)
    da2 = df * forgets * (1-forgets)
    da3 = do * outputs * (1-outputs)
    da4 = dg * (1 - gate**2)

    dx = np.dot(da1, Wx[:,0:H].T) + np.dot(da2, Wx[:,H:2*H].T)\
         + np.dot(da3, Wx[:,2*H:3*H].T) + np.dot(da4, Wx[:,3*H:].T)
    dprev_h = np.dot(da1, Wh[:,0:H].T) + np.dot(da2, Wh[:,H:2*H].T) \
              + np.dot(da3, Wh[:,2*H:3*H].T) + np.dot(da4, Wh[:,3*H:].T)
    dWx[:,:H] = np.dot(x.T, da1)
    dWx[:,H:2*H] = np.dot(x.T, da2)
    dWx[:,2*H:3*H] = np.dot(x.T, da3)
    dWx[:,3*H:] = np.dot(x.T, da4)
    dWh[:,:H] = np.dot(prev_h.T, da1)
    dWh[:,H:2*H] = np.dot(prev_h.T, da2)
    dWh[:,2*H:3*H] = np.dot(prev_h.T, da3)
    dWh[:,3*H:] = np.dot(prev_h.T, da4)
    db[:H] = np.sum(da1, axis=0)
    db[H:2*H] = np.sum(da2, axis=0)
    db[2*H:3*H] = np.sum(da3, axis=0)
    db[3*H:] = np.sum(da4, axis=0)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    T vectors
    dimension D
    hidden size of H
    minibatch containing N sequences

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    h = np.zeros((N,T,H))
    cache = {}

    prev_h = h0
    prev_c = np.zeros((N,H))
    for t in range(T):
        next_h, next_c, cache[t] = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)
        h[:,t,:] = next_h

        prev_c = next_c
        prev_h = next_h

    cache = cache, D

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    cache, D = cache
    dnext_h = np.zeros((N, H))
    dnext_c = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))

    for t in reversed(range(T)):
        dnext_h = dnext_h + dh[:,t,:]

        dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt = \
            lstm_step_backward(dnext_h, dnext_c, cache[t])

        dWx += dWxt
        dWh += dWht
        db += dbt

        dnext_h = dprev_h
        dnext_c = dprev_c

    dh0 = dnext_h

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    # 生成score

    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
