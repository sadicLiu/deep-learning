# TensorFlow用过的一些API

- [TensorFlow用过的一些API](#tensorflow%E7%94%A8%E8%BF%87%E7%9A%84%E4%B8%80%E4%BA%9Bapi)
    - [`tf.get_variable`](#tfgetvariable)
    - [`tf.nn.avg_pool`](#tfnnavgpool)
    - [`tf.pad`](#tfpad)
    - [`tf.nn.moments`](#tfnnmoments)
    - [`tf.nn.batch_normalization`](#tfnnbatchnormalization)
    - [`tf.nn.softmax_cross_entropy_with_logits_v2`](#tfnnsoftmaxcrossentropywithlogitsv2)
    - [`tf.trainable_variables`](#tftrainablevariables)
    - [`tf.nn.l2_loss`](#tfnnl2loss)
    - [`tf.add_n`](#tfaddn)
    - [`tf.gfile.Glob`](#tfgfileglob)
    - [`tf.sparse_to_dense`](#tfsparsetodense)
    - [`tf.train.SummarySaverHook`](#tftrainsummarysaverhook)
    - [`tf.train.LoggingTensorHook`](#tftrainloggingtensorhook)
    - [`tf.train.MonitoredTrainingSession`](#tftrainmonitoredtrainingsession)
    - [`tf.read_file`](#tfreadfile)
    - [`tf.image.decode_image`](#tfimagedecodeimage)
    - [`tf.contrib.layers.flatten`](#tfcontriblayersflatten)
    - [`tf.nn.l2_normalize`](#tfnnl2normalize)
    - [`collection`](#collection)
    - [`tf.nn.sparse_softmax_cross_entropy_with_logits`](#tfnnsparsesoftmaxcrossentropywithlogits)
    - [`tf.argmax`](#tfargmax)
    - [`tf.control_dependencies`](#tfcontroldependencies)
    - [`minimize、compute_gradients、apply_gradients`](#minimizecomputegradientsapplygradients)
    - [`slim.repeat`](#slimrepeat)
    - [`tf.image.sample_distorted_bounding_box`](#tfimagesampledistortedboundingbox)
    - [`tf.where`](#tfwhere)
    - [links](#links)

## `tf.get_variable`

## `tf.nn.avg_pool`

## `tf.pad`

## `tf.nn.moments`

```Python
tf.nn.moments(
    x,
    axes,
    shift=None,
    name=None,
    keep_dims=False
)
```

- 求均值方差的
- for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2]
- for simple batch normalization pass axes=[0] (batch only).

## `tf.nn.batch_normalization`

```Python
tf.nn.batch_normalization(
    x,
    mean,
    variance,
    offset(beta),
    scale(gamma),
    variance_epsilon,
    name=None
)
```

- mean, variance, beta, gamma具有相同的维度（等于x的depth）

## `tf.nn.softmax_cross_entropy_with_logits_v2`

```Python
tf.nn.softmax_cross_entropy_with_logits_v2(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

- Returns: A 1-D Tensor of length batch_size of the same type as logits with the softmax cross entropy loss.

## `tf.trainable_variables`

```Python
tf.trainable_variables(scope=None)
```

- Returns: A list of Variable objects.

## `tf.nn.l2_loss`

- Computes half the L2 norm of a tensor without the sqrt: `output = sum(t ** 2) / 2`

## `tf.add_n`

- Adds all input tensors element-wise.

## `tf.gfile.Glob`

- Returns a list of files that match the given pattern(s).

## `tf.sparse_to_dense`

- 把label转成one-hot
- demo
    ```Python
        import tensorflow as tf
        import numpy as np

        tf.InteractiveSession()
        indices = np.array(range(10)).reshape([10, 1])
        # print(indices)

        lables = np.array([0, 3, 2, 1, 1, 5, 6, 4, 1, 2]).reshape([10, 1])
        # print(lables.shape)

        y = tf.concat(values=[indices, lables], axis=1)
        print(y.eval())
        y = tf.sparse_to_dense(y, [10, 10], 1.0, 0.0)
        print(y.eval())
    ```

## `tf.train.SummarySaverHook`

- Save summaries every N steps
- [doc](https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/train/SummarySaverHook)
- resnet_cifar->resnet_main.py(67)

## `tf.train.LoggingTensorHook`

- Prints the given tensors every N local steps, every N seconds, or at end
- resnet_cifar->resnet_main.py(76)

## `tf.train.MonitoredTrainingSession`

- 在run过程中的集成一些操作，比如输出log，保存summary等

## `tf.read_file`

```Python
tf.read_file(
    filename,
    name=None
)
```

- Reads and outputs the entire contents of the input filename.

## `tf.image.decode_image`

```Python
tf.image.decode_image(
    contents,
    channels=None,
    name=None
)
```

- Convenience function for decode_bmp, decode_gif, decode_jpeg, and decode_png.
- Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type uint8.

## `tf.contrib.layers.flatten`

```Python
tf.contrib.layers.flatten(
    inputs,
    outputs_collections=None,
    scope=None
)
```

- Assumes that the first dimension represents the batch.
- Returns: A flattened tensor with shape [batch_size, k].

## `tf.nn.l2_normalize`

```Python
tf.nn.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
```

- output = x / sqrt(max(sum(x**2), epsilon))
- Returns: A Tensor with the same shape as x.

## `collection`

- demo: [Collection](./0300_TF_Collection.ipynb)

## `tf.nn.sparse_softmax_cross_entropy_with_logits`

- 这个输入的label是一个值,不用one-hot编码

## `tf.argmax`

- 返回最大的那个值所在的下标

## `tf.control_dependencies`

- demo: [Dependency](./0400_Dependency.ipynb)

## `minimize、compute_gradients、apply_gradients`

- minimize = compute_gradients + apply_gradients
- The difference it that: if you use the separated functions( tf.gradients, tf.apply_gradients), you can apply other mechanism between them, such as gradient clipping

## `slim.repeat`

- 假如要在tensorflow中定义三个卷积层：

   ```Python
    net = ...
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

   ```

- 使用slim.repeat的写法：

    ```Python
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    ```

## `tf.image.sample_distorted_bounding_box`

- Generate a single randomly distorted bounding box for an image.

## `tf.where`

- api
  
    ```tensorflow
    tf.where(
        condition,
        x=None,
        y=None,
        name=None
    )
    ```

- 解析：condition是一个掩码矩阵，condition、x、y的shape相同，condition=True的地方从x中取值，condition=False的地方从y中取值
  
    ```tensorflow
    mask = [True, False, False]
    a = [1,1,1]
    b = [0,0,0]

    a = tf.where(mask, a, b)
    print(a.eval()) # [1, 0, 0]
    ```
  
## links

- [normalization](https://www.tensorflow.org/api_guides/python/nn#Normalization)
- [padding](https://www.jianshu.com/p/05c4f1621c7e)