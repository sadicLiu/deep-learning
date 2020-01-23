import tensorflow as tf

"""
from_tensor_slices这个函数是把数组的第一维当成是样本数
"""

a = tf.random_uniform([4, 10])
print(a)
sess = tf.InteractiveSession()
print(sess.run(a))

dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10, 10)"

dataset2 = tf.contrib.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"