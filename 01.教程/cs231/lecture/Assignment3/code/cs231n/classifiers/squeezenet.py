import tensorflow as tf

NUM_CLASSES = 1000


def fire_module(x,inp,sp,e11p,e33p):
    '''
    A Fire module is comprised of:
        a squeeze convolution layer(which has only 1x1 filters),
        feeding into an expand layer that has a mix of 1x1 and 3x3 convolution filters
    '''

    # sp is the number of filters in the squeeze layer (all 1x1)
    # e11p is the number of 1x1 filters in the expand layer
    # e33p is the number of 3x3 filters in the expand layer

    with tf.variable_scope("fire"):
        with tf.variable_scope("squeeze"):
            # inp是上一层输入（这里是x）的channel数，sp是这一层filter的depth
            W = tf.get_variable("weights",shape=[1,1,inp,sp])
            b = tf.get_variable("bias",shape=[sp])
            s = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")+b
            s = tf.nn.relu(s)
        with tf.variable_scope("e11"):
            W = tf.get_variable("weights",shape=[1,1,sp,e11p])
            b = tf.get_variable("bias",shape=[e11p])
            e11 = tf.nn.conv2d(s,W,[1,1,1,1],"VALID")+b
            e11 = tf.nn.relu(e11)
        with tf.variable_scope("e33"):
            W = tf.get_variable("weights",shape=[3,3,sp,e33p])
            b = tf.get_variable("bias",shape=[e33p])
            e33 = tf.nn.conv2d(s,W,[1,1,1,1],"SAME")+b
            e33 = tf.nn.relu(e33)
        return tf.concat([e11,e33],3)


class SqueezeNet(object):
    def extract_features(self, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        with tf.variable_scope('features', reuse=reuse):
            # (N, 111, 111, 64)
            with tf.variable_scope('layer0'):
                W = tf.get_variable("weights",shape=[3,3,3,64])
                b = tf.get_variable("bias",shape=[64])
                x = tf.nn.conv2d(x,W,[1,2,2,1],"VALID")
                x = tf.nn.bias_add(x,b)
                layers.append(x)
            # (N, 111, 111, 64)
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            # (N, 55, 55, 64)
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            # (N, 55, 55, 128)
            with tf.variable_scope('layer3'):
                # s.shape: (N, 55, 55, 16)
                # e11.shape: (N, 55, 55, 64)
                # e33.shape: (N, 55, 55, 64)
                x = fire_module(x,64,16,64,64)
                layers.append(x)
            # (N, 55, 55, 128)
            with tf.variable_scope('layer4'):
                x = fire_module(x,128,16,64,64)
                layers.append(x)
            # (N, 27, 27, 128)
            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            # (N, 27, 27, 256)
            with tf.variable_scope('layer6'):
                x = fire_module(x,128,32,128,128)
                layers.append(x)
            # (N, 27, 27, 256)
            with tf.variable_scope('layer7'):
                x = fire_module(x,256,32,128,128)
                layers.append(x)
            # (N, 13, 13, 256)
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            # (N, 13, 13, 384)
            with tf.variable_scope('layer9'):
                x = fire_module(x,256,48,192,192)
                layers.append(x)
            # (N, 13, 13, 384)
            with tf.variable_scope('layer10'):
                x = fire_module(x,384,48,192,192)
                layers.append(x)
            # (N, 13, 13, 512)
            with tf.variable_scope('layer11'):
                x = fire_module(x,384,64,256,256)
                layers.append(x)
            # (N, 13, 13, 512)
            with tf.variable_scope('layer12'):
                x = fire_module(x,512,64,256,256)
                layers.append(x)
        return layers

    def __init__(self, save_path=None, sess=None):
        """Create a SqueezeNet model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        - input: optional input to the model. If None, will use placeholder for input.
        """
        self.image = tf.placeholder('float',shape=[None,None,None,3],name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')
        self.layers = []
        x = self.image
        self.layers = self.extract_features(x, reuse=False)

        # (N, 13, 13, 512)
        self.features = self.layers[-1]
        with tf.variable_scope('classifier'):
            # (N, 13, 13, 512)
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            # (N, 13, 13, 1000)
            with tf.variable_scope('layer1'):
                W = tf.get_variable("weights",shape=[1,1,512,1000])
                b = tf.get_variable("bias",shape=[1000])
                x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
                x = tf.nn.bias_add(x,b)
                self.layers.append(x)
            # (N, 13, 13, 1000)
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            # (N, 1, 1, 1000)
            with tf.variable_scope('layer3'):
                x = tf.nn.avg_pool(x,[1,13,13,1],strides=[1,13,13,1],padding='VALID')
                self.layers.append(x)

        # 直接reshape，没有全连接层
        # (N, 1000)
        self.classifier = tf.reshape(x,[-1, NUM_CLASSES])

        if save_path is not None:
            # saver = tf.train.Saver()
            # saver.restore(sess, save_path)
            saver = tf.train.import_meta_graph(save_path + 'squeezenet.ckpt.meta')
            saver.restore(sess,save_path + 'squeezenet.ckpt')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier), name='loss')
