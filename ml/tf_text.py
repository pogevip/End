# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# learn_rate = 0.5
# epochs = 10
# batch_size = 100
#
# #输入输出
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
#
# #内部参数
# W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
# b1 = tf.Variable(tf.random_normal([300]), name='b1')
#
# W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
# b2 = tf.Variable(tf.random_normal([10]), name='b2')
#
# #运算
# hidden = tf.nn.relu(tf.add(tf.multiply(x, W1), b1))
# y_ = tf.nn.softmax(tf.add(tf.multiply(hidden, W2), b2))
#
# #平滑y_
# y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
#
# #计算交叉熵
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1-y_clipped)), axis=1)
#
# #优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)
#
# #初始化算子
# init_op = tf.global_variables_initializer()
#
# #创建准确率结点
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     writer = tf.summary.FileWriter("logs/", sess.graph)
#
#     total_batch = int(len(mnist.train.labels)/batch_size)
#
#     for epoch in range(epochs):
#         avg_cost=0
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
#             _, c = sess.run([optimizer, cross_entropy], feed_dict={x:batch_x, y:batch_y})
#             avg_cost += c / total_batch
#
#             print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
#
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

import tensorflow as tf
import numpy as np

# input_data = tf.Variable(np.random.rand(10, 30, 200), dtype=np.float32)
# input_data = tf.expand_dims(input_data, -1)
# print('tf.nn.input : ', input_data)
# pool_out=list()
# for filter_size in [2,3,4]:
#     with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
#         W = tf.Variable(tf.truncated_normal([filter_size, 200, 1, 3], stddev=0.1),
#                         name='W')
#         b = tf.Variable(tf.constant(0.1, shape=[3]), name='b')
#         conv = tf.nn.conv2d(
#             input_data, W,
#             strides=[1, 1, 1, 1],
#             padding='VALID',
#             name='conv'
#         )
#         print('tf.nn.pool : ', conv)
#         h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
#         pooled = tf.nn.max_pool(
#             h,
#             ksize=[1, 30 - filter_size + 1, 1, 1],
#             strides=[1, 1, 1, 1],
#             padding='VALID',
#             name='pool',
#         )
#         print('tf.nn.pool : ', pooled)
#         pool_out.append(pooled)
#
# h_pool = tf.concat(pool_out, 3)
# print('tf.nn.h_pool : ', h_pool)
# h_pool_flat = tf.reshape(h_pool, [-1, 9])
# print('tf.nn.hp_flat : ', h_pool_flat)

# def lenth(sequences):
#     used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
#     seq_len = tf.reduce_sum(used, reduction_indices=1)
#     return tf.cast(seq_len, tf.int32)
#
# l = [
# [[0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0,0,0]],
# [[0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0,0,0]],
# [[0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0,0,0], [0,0,0]],
# [[0.1,0.2,0.3], [0.1,0.2,0.3], [0,0,0], [0,0,0], [0,0,0]],
# [[0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0,0,0]],
# ]
#
# with tf.Session() as sess:
#     print(sess.run(tf.reduce_max(tf.abs(l), reduction_indices=2)))

import numpy as np
# from tensorflow.contrib import learn
#
# x_text = [['I love you', 'This must be boy', 'This is a a dog'],
#           ['This is a cat','hao high ou', 'wo ai bei jing tian an men']]
#
#
# max_document_length = max([len(x) for x in x_text])
# max_sentence_length = np.max([[len(x.split(' ')) for x in doc] for doc in x_text])
#
# tmp = np.array(x_text).reshape(-1)
#
# ## Create the vocabularyprocessor object, setting the max lengh of the documents.
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
# vocab_processor.fit(tmp)
#
# ## Transform the documents using the vocabulary.
# res = []
# for doc in x_text:
#     x = np.array(list(vocab_processor.fit_transform(doc)))
#     res.append(np.pad(x, ((0, max_document_length-len(x)), (0,0)), 'constant'))
# res = np.array(res)
# print(res)
#
# ## Extract word:id mapping from the object.
# vocab_dict = vocab_processor.vocabulary_._mapping
#
# ## Sort the vocabulary dictionary on the basis of values(id).
# ## Both statements perform same task.
# #sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
# sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
#
# ## Treat the id's as index into list and create a list of words in the ascending order of id's
# ## word with id i goes at index i of the list.
# vocabulary = list(list(zip(*sorted_vocab))[0])
#
# print(vocab_dict)
# print(len(vocab_processor.vocabulary_))
# print(len(vocab_dict))


import tensorflow as tf
from tensorflow.contrib import rnn

tf.reset_default_graph()
# 创建输入数据
X = np.random.randn(2, 4, 5)# 批次 、序列长度、样本维度

# 第二个样本长度为3
X[1,2:] = 0

print(X)

def lenth(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


GRU_cell_fw = rnn.GRUCell(3)
GRU_cell_bw = rnn.GRUCell(3)
((fw_out, bw_out), (_,_)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                            cell_bw=GRU_cell_bw,
                                                            inputs=X,
                                                            sequence_length=lenth(X),
                                                            dtype=tf.float64)
outputs = tf.concat((fw_out, bw_out), axis=2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

out=sess.run([outputs])
print(out)