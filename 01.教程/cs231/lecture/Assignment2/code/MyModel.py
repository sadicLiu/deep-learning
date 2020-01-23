import tensorflow as tf
import numpy as np
import math
import time


import argparse
import sys

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

def batch_norm(x, is_training, decay = 0.999, epsilon=1e-4):

    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))

    scale = tf.Variable(tf.ones(params_shape))
    beta = tf.Variable(tf.zeros(params_shape))
    pop_mean = tf.Variable(tf.zeros(params_shape), trainable=False)
    pop_var = tf.Variable(tf.ones(params_shape), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(x, axis)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        # 有了这句话，[train_mean, train_var]会在return之前执行
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

FLAGS = None

def train():
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X-input")
        y = tf.placeholder(tf.int64, [None], name="y-input")
        is_training = tf.placeholder(tf.bool, name='is_training-input')
        tf.summary.image("input", X, 10)

    training_mode = is_training is not None

    # bn-conv(32)-pool #1
    with tf.name_scope("layer1"):
        bn = batch_norm(X, is_training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    # bn-conv(48)-pool #2
    with tf.name_scope('layer2'):
        bn = batch_norm(pool, is_training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=48,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    # bn-conv(64)-pool #2
    with tf.name_scope('layer3'):
        bn = batch_norm(pool, is_training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    # dense1 1024 output units
    with tf.name_scope('dense1'):
        pool_flat = tf.reshape(pool, [-1, 4 * 4 * 64])
        dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
        tf.summary.histogram('dense', dense)

    # dropout
    with tf.name_scope('dropout'):
        dropout = tf.layers.dropout(
            inputs=dense, rate=FLAGS.dropout, training=training_mode)

    # dense2 10 output units
    with tf.name_scope('dense2'):
        y_out = tf.layers.dense(inputs=dropout, units=10)
        tf.summary.histogram('dense2', dense)

    # cross entropy loss
    with tf.name_scope('loss'):
        diff = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=y)
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # train
    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(FLAGS.start_learning_rate, global_step,
                                                   100, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(cross_entropy, global_step)

    # calculate predict accuracy of the model
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(y, tf.argmax(y_out, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # write summary to disk
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
    tf.global_variables_initializer().run()

    # start training

    # shuffle indicies
    saver = tf.train.Saver()

    train_indicies = np.arange(X_train.shape[0])
    np.random.shuffle(train_indicies)

    iter_per_epoch = int(math.ceil(X_train.shape[0]/FLAGS.batch_size))
    for e in range(FLAGS.epochs):
        start_time = time.time()
        for i in range(iter_per_epoch):
            # 每100次迭代，测试并记录模型在验证集上的准确率
            if i % 100 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={X:X_val, y:y_val, is_training:False})
                validation_writer.add_summary(summary, global_step=e*iter_per_epoch+i)
                print('Validation Accuracy at step %s: %s' % (e*iter_per_epoch+i, acc))
            else:
                # generate indicies for the batch
                # 取模是因为上面是上取整，有可能超出总样本数
                start_idx = (i*FLAGS.batch_size)%X_train.shape[0]
                idx = train_indicies[start_idx:start_idx+FLAGS. batch_size]

                summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={X: X_train[idx,:],
                               y: y_train[idx],
                               is_training: True }
                )

                train_writer.add_summary(summary, global_step=e*iter_per_epoch+i)

        duration = time.time() - start_time
        print('The time span of 1 epoch: %s' % (duration))
        saver.save(sess, save_path=FLAGS.log_dir+'/model', global_step=(e+1)*iter_per_epoch)

    # test the model
    acc = sess.run(accuracy, feed_dict={X:X_test, y:y_test, is_training:False})
    print('Test Accuracy: %s' % (acc))

    # 如果把这两个writer关上，在TensorBoard中显示不出来
    # train_writer.close()
    # validation_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        # Deletes everything under dirname recursively.
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of batch size.')
    parser.add_argument('--start_learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/cs231n/2',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
