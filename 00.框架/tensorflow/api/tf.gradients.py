import tensorflow as tf
import numpy as np

"""
使用tf.gradients反向求导
这个demo是比较正常的，直接从最后结果向前传播
"""

# y = w*x
w = tf.Variable([[1,2]])
x = [[3],[4]]
y = tf.matmul(w, x)

# z = u*y
u = tf.Variable([[5]])
# z = tf.matmul(u, y)
z = 5*y

# dy/dw
dyw = tf.gradients(ys=y, xs=w)

# dz/dw
dzw = tf.gradients(ys=z, xs=w)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

result = sess.run(y)
print(result)

dyw = sess.run(dyw)
print(dyw)

dzw = sess.run(dzw)
print(dzw)
