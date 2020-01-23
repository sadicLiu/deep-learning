# TensorFlow Learning Notes

- [Official Tutorial][318ec1de]

  [318ec1de]: https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial "Official Tutorial"


## Part1_GetStart
> TensorFlow的一些基本用法

### 0100_Install
>	安装TensorFlow

- 参考该文件夹下的文档基本就可以搞定
- [可以参考这篇博客][1590c02c]

  [1590c02c]: http://blog.csdn.net/chongtong/article/details/53905625?locationNum=6&fps=1 "博客"

### 0200_GetStart
> TensorFlow入门的一些demo，写在了jupyter里

- 0100_BasicDemo

	tensorflow的一些基本语法

- 0200_LinearModel

	使用tf的constant、Variable、placeholder等自己定义一个线性模型，并用tf.train API训练此模型

- 0300_tf.contrib.learn

	TensorFlow中有两种级别的API：

	1. TensorFlow Core：允许你进行完全的编程控制
	2. higher level API(方法名中包含contrib的):这些API是构建在TensorFlow Core之上的，你可以直接使用内置的模型

	这个demo中，直接使用内置的`Logistic Regression`模型创建一个`estimator`,并用其`fit`、`evaluate`方法进行训练和测试

- 0400_CUstomModel

	同样是使用higher level API,不同的是用`tf.contrib.learn.Estimator`构造一个自定义的model


## Part2_MNIST
> 使用TensorFlow分类MNIST数据集

### 0300_MnistForBeginner

- 0100_MNISTForBeginner

	自定义一个线性模型，然后用`tf.model.softmax`分类MNIST

### 0400_MnistForExpert

- 0100_MNISTForExpert

	自定义`softmax`分类数据

- 0200_ConvNetModel

	两层卷积，两层全连接

### 0500_TensorFlowMechanics101

- 0100_tf.name_scope

	`name_scope`的一个demo，这个东西就类似于局部变量


## Part3_high-level_API
> TensorFlow high-level API的demo

### 0600_tf.contrib.learnQuickstart

- 0100_NN

	使用`tf.contrib.learn`API构建一个神经网络，分类iris数据

### 0700_BuildingInputFunctions

- 0100_HouseingPriceRegression

    定义一个`input_fn`，并用lambda表达式传给`classifier.fit`，实现不同数据集共用一个函数

**上面两个demo，一个是用`tf.contrib.learn.datasets.base.load_csv_with_header`导入数据，另一个是用`pandas.read_csv`导入数据**

### 0800_LoggingandMonitoringBasics

- 0100_NN

    这个文件的原始版本与0600中的文件一样，是在此文件基础上的改动

    ValidationMonitor的使用，主要包括：

    1. 用ValidationMonitor每隔N步在测试集上验证一次准确率
    2. ValidationMonitor的使用依赖于checkpoints,初始化classifier时用`config = tf.contrib.learn.RunConfig(save_checkpoints_secs=1)`配置tf每1秒保存一下checkpoints
    3. 可以自定义log中显示的validation_metrics
    4. 让训练提前终止

    使用TensorBoard可视化保存的结果

## Part4_TensorBoard
> 使用TensorBoard做一些可视化

### 0900_VisualizingLearning

- `mnist_with_summaries.py`是一个使用TensorBoard的demo

### 1000_EmbeddingVisualization

- 这个目前用不到

### 1100_GraphVisualization

- 这里面主要讲了`TensorBoard`中`Graph`的一些用法，还是很有用的

### 1200_HistogramDashboard

- 0100_Histogram.py

    使用`tf.summary.histogram`记录生成的高斯分布

- 0200_MultiModel.py

    把两个高斯分布合并并记录

## Part5_TFDetails
> TensorFlow's Details

### 1300_Variables

- 模型的存储与恢复，这个挺有用的

### 1400_Tensor

- tensor介绍

### 1500_SharingVariables

- 这个没怎么看懂

### 1600_ThreadingAndQueues

- 线程、队列、异常处理等内容，目前用不上

### 1700_ReadingData

- demo1

- demo2_DatasetAPI

    dataset api 一些demo
    
- demo3_batch

    1. tf.train.shuffle_batch
    2. tf.train.batch
    
### 2100_Losses

- demo1: 利用Tensorflow Collection保存loss

- demo2: tf.losses.add_loss、tf.get_collection(tf.GraphKeys.LOSSES)


