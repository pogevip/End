#coding=utf-8

import os, time, pickle
import tensorflow as tf

from TextClassifier.HAN import HAN
from TextClassifier.HAN_gen_tfrecord import TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_DICT_PATH


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate (default: 0.01)")


FLAGS = tf.flags.FLAGS


with open('data/train_data/HAN_data_size.config', 'rb') as fp:
    data_size_config = pickle.load(fp)


def decode_from_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'input_raw': tf.FixedLenFeature([], tf.int64),
                                       })  # 取出包含input_raw和label的feature对象
    input_raw = tf.reshape(features['input_raw'], [data_size_config['MAX_DOC_LEN'], data_size_config['MAX_SEN_LEN']])
    label = features['label']
    return input_raw, label


def batch_reader(filenames, batch_size, thread_count, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    input_raw, label = decode_from_tfrecord(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + thread_count * batch_size
    x_batch, y_batch = tf.train.shuffle_batch([input_raw, label],
                                              batch_size=batch_size,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue,
                                              num_threads=thread_count)
    return x_batch, y_batch


def train(train_file_path, dev_file_path, vocab_dic_path):

    # get all dev data
    dev_x, dev_y = decode_from_tfrecord(dev_file_path)

    # get vocab size
    with open(vocab_dic_path, 'rb') as fp:
        vocab_dict = pickle.load(fp)
        vocab_size = len(vocab_dict)

    with tf.Session() as sess:
        han = HAN(
            vocab_size=vocab_size,
            num_classes=dev_y.shape[1],
            embedding_size=FLAGS.embedding_size,
            hidden_size=FLAGS.hidden_size)


        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        #RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(han.loss, tvars), FLAGS.grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g,v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar('loss', han.loss)
        acc_summary = tf.summary.scalar('acc', han.acc)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        train_x, train_y = batch_reader([train_file_path], FLAGS.batch_size, 3)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.max_sentence_num: 30,
                han.max_sentence_length: 30,
                han.batch_size: 64
            }
            _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, han.loss, han.acc],
                                                          feed_dict)

            time_str = str(int(time.time()))
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return step

        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.max_sentence_num: 30,
                han.max_sentence_length: 30,
                han.batch_size: 64
            }
            step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, han.loss, han.acc], feed_dict)
            time_str = str(int(time.time()))
            print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost,
                                                                                               accuracy))
            if writer:
                writer.add_summary(summaries, step)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in range(FLAGS.num_epochs):
                x_batch, y_batch = sess.run([train_x, train_y])
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(dev_x, dev_y, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
        except tf.errors.OutOfRangeError:
            print("Trainning finished!")
        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()


def main(argv=None):
    train(TRAIN_DATA_PATH, DEV_DATA_PATH, VOCAB_DICT_PATH)



if __name__ == '__main__':
    tf.app.run()


