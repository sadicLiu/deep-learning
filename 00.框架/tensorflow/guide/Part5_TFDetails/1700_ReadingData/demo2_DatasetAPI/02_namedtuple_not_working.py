import tensorflow as tf

"""
This pieces of code not working
"""

dataset = tf.contrib.data.Dataset.from_tensor_slices(
    {"a": tf.random_uniform([4]),
     "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})

print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
