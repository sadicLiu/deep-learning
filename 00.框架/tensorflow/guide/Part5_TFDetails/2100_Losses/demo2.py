import tensorflow as tf
import re


with tf.name_scope('my_loss1') as scope:
    v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1))
    v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
    tf.losses.add_loss(v1 + v2)
with tf.name_scope('my_loss2') as scope:
    v3 = tf.get_variable(name='v3', shape=[1], initializer=tf.constant_initializer(3))
    v4 = tf.get_variable(name='v4', shape=[1], initializer=tf.constant_initializer(4))
    tf.losses.add_loss(v3 + v4)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    loss = tf.get_collection(tf.GraphKeys.LOSSES)

    print(loss)
    print(tf.get_collection(tf.GraphKeys.LOSSES, scope='my_loss1'))
    print(tf.get_collection(tf.GraphKeys.LOSSES, scope='my_loss2'))
    print(loss[0].op.name)
    print(loss[1].op.name)

    loss_name = re.sub('%s[0-9]*/' % 'my_loss', '', loss[0].op.name)
    print(loss_name)
