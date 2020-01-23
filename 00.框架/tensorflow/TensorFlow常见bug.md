# 下面总结了一些tf中容易出现的一些bug

1. 进行reduce操作的时候没有设置`keepdims=True`

	```
		ValueError: Dimensions must be equal, but are 5 and 64 for 'truediv' (op: 'RealDiv') with input shapes: [64,5], [64].
	```

	```
		# 正确代码：
		x_sum = tf.reduce_sum(x_exp, axis=1, keep_dims=True)
	```

2. 提示类型不一致，通常是在定义变量的时候没有指定好类型

	```
		# np定义数组的时候就指定astype
		b = tf.Variable(np.zeros([1, self.config.n_classes]).astype(np.float32), name='biases')
	```

3. tf.reshape(x, shape), 这个shape不能是`[None, ...]`的形式，可以写成`[-1, ...]`, 维数必须是可以明确计算的

4. tensorflow定义一个数，`shape=[]`，如果弄成1维的，会有如下提示：
	```
		ValueError: Shapes (1,) and () are incompatible
	```

	```
		self.dropout_placeholder = tf.placeholder(tf.float32, shape=[])
	```

5. tensorflow中作为placeholder的索引值必须是int
	```
		images_placeholder: X_train[indices, :]

		IndexError: arrays used as indices must be of integer (or boolean) type
	```

	```
		indices = np.zeros([800])		# 错误
		indices = np.zeros([800], dtype=np.int)		# 正确
	```
