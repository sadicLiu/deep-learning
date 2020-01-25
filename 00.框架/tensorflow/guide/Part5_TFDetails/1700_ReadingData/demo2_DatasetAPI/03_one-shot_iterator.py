import tensorflow as tf

sess = tf.InteractiveSession()

"""
整个dataset只能遍历一次
"""
dataset = tf.contrib.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    assert i == value
