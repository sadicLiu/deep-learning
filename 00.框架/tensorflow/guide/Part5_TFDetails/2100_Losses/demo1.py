import tensorflow as tf

'''
    tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
    tf.get_collection：从一个结合中取出全部变量，是一个列表
    tf.add_n：把一个列表的东西都依次加起来
'''

v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    loss = tf.get_collection('loss')
    print(loss.op.name)
    print(loss)
    print(sess.run(tf.get_collection('loss')[0]))
    print(sess.run(tf.get_collection('loss')[1]))
    print(sess.run(tf.add_n(tf.get_collection('loss'))))
