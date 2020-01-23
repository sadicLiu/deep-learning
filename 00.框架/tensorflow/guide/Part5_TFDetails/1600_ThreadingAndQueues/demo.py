import threading
import numpy as np
import tensorflow as tf


# 创建一个函数实现多线程，参数为Coordinater和线程号
def func(coord, t_id):
    count = 0
    while not coord.should_stop():  # 不应该停止时计数
        print('thread ID:', t_id, 'count =', count)
        count += 1

        # 第一个进程跑的太快了，如果设置的数太小，有可能其他进程还没开始就已经结束了，看不出效果
        if (count == 50):  # 计到50时请求终止
            coord.request_stop()


coord = tf.train.Coordinator()
threads = [threading.Thread(target=func, args=(coord, i)) for i in range(4)]
# 开始所有线程
for t in threads:
    t.start()

coord.join(threads)  # 等待所有线程结束
