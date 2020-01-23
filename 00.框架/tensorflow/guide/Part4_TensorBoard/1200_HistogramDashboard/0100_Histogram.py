import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter("/tmp/histogram_example")
merged = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
    k_val = step/float(N)
    summaries = sess.run(merged, feed_dict={k: k_val})
    writer.add_summary(summaries, global_step=step)