import tensorflow as tf
import numpy as np

# 这个demo是从任意位置向前传播

"""
    score = Wx
    dW = dscore * x
    
    W: (2,2)
    x: (2,1) 两条记录
    score: (2,1)
    
    现在如果是只对score[0]感兴趣，把dscore设成[1.0, 0], 用这个反向求导
"""

x = tf.constant([[1],[2]])
W = tf.Variable(initial_value=[[3,4],[5,6]])
scores = tf.matmul(W, x)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(scores))  # [11,17].T


'''
[array([[1],
       [1]])]
'''
dscores = tf.gradients(xs=scores, ys=scores)
print(sess.run(dscores))


'''
[array([[1, 2],
       [1, 2]])]
'''
dW = tf.gradients(xs=W, ys=scores)
print(sess.run(dW))


'''
[array([[1, 2],
       [0, 0]])]
'''
dscore0 = np.array([[1],[0]])
dW0 = tf.gradients(xs=W, ys=scores, grad_ys=dscore0)
print(sess.run(dW0))

