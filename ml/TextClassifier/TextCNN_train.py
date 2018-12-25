#coding:utf-8

import tensorflow as tf
import os
import time
import pickle
import datetime
import numpy as np
from TextClassifier.TextCNN import TextCNN
import random
from TextClassifier.data_helper import stuff_doc, gen_one_hot, read_data

# Parameters
# ==================================================
tf.flags.DEFINE_string("data_dir", "../data/trainSet/TextCnn", "data directory")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


FLAGS = tf.flags.FLAGS


data_option = 0

def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = list(range(data_size))
            random.shuffle(shuffle_indices)

            shuffled_x = [x[i] for i in shuffle_indices]
            shuffled_y = [y[i] for i in shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            train_x_tmp = shuffled_x[start_index:end_index]
            train_y_tmp = shuffled_y[start_index:end_index]

            train_x = np.array(list(map(lambda doc: stuff_doc(doc, model_option=1, data_option=data_option), train_x_tmp)))
            train_y = gen_one_hot(train_y_tmp)

            yield train_x, train_y


def train(train_file_path, dev_file_path, vocab_dic_path):
    # Training
    # ==================================================

    # get all dev data
    dev_x, dev_y = read_data(dev_file_path)
    dev_x = np.array(list(map(lambda doc: stuff_doc(doc), dev_x)))
    dev_y = gen_one_hot(dev_y)

    train_x, train_y = read_data(train_file_path)

    # get vocab size
    with open(vocab_dic_path, 'rb') as fp:
        vocab_dict = pickle.load(fp)
        vocab_size = len(vocab_dict)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                max_sequence_lenth = dev_x.shape[1],
                vocab_size = vocab_size,
                embedding_size = FLAGS.embedding_dim,
                class_num = dev_y.shape[1],
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                filter_num = FLAGS.num_filters,
                l2_reg = FLAGS.l2_reg_lambda)


            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.acc)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

                batches = batch_iter(train_x, train_y, FLAGS.batch_size, FLAGS.num_epochs)

                for x_batch, y_batch in batches:
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(dev_x, dev_y, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    if data_option == 0:
        p = 'rough'
    else:
        p = 'rigour'
    train_data_path = os.path.join(FLAGS.data_dir, p, 'train')
    val_data_path = os.path.join(FLAGS.data_dir, p, 'val')
    vacab_data_path = '../data/trainSet/vacab.dic'
    train(train_data_path, val_data_path, vacab_data_path)


if __name__ == '__main__':
    tf.app.run()