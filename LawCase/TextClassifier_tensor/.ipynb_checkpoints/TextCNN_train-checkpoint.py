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
from TextClassifier.data_helper import MAX_DOC_LEN_ROUGH, MAX_DOC_LEN_RIGOUR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Parameters
# # ==================================================
# tf.flags.DEFINE_string("data_dir", "../data/trainSet/TextCnn", "data directory")
#
# # Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
#
# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
#
#
# FLAGS = tf.flags.FLAGS
#
# data_dir = FLAGS.data_dir
# embedding_dim = FLAGS.embedding_dim
# filter_sizes = FLAGS.filter_sizes
# num_filters = FLAGS.num_filters
# dropout_keep_prob = FLAGS.dropout_keep_prob
# l2_reg_lambda = FLAGS.l2_reg_lambda
# batch_size = FLAGS.batch_size
# num_epochs = FLAGS.num_epochs
# evaluate_every = FLAGS.evaluate_every
# checkpoint_every = FLAGS.checkpoint_every
# num_checkpoints = FLAGS.num_checkpoints

data_dir = "../data/trainSet/TextCnn"
embedding_dim = 128
filter_sizes = '3,4,5'
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.2
batch_size = 256
num_epochs = 10
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5

data_option = 1
p = 'rough' if data_option == 0 else 'rigour'
max_sequence_lenth = MAX_DOC_LEN_ROUGH if data_option==0 else MAX_DOC_LEN_RIGOUR


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


def train(train_file_path, val_file_path, vocab_dic_path):
    # Training
    # ==================================================

    # get all dev data

    val_res = {
        'loss' : [],
        'acc' : [],
    }
    timestamp = None

    dev_x, dev_y = read_data(val_file_path)
    dev_x = np.array(list(map(lambda doc: stuff_doc(doc, model_option=1, data_option=data_option), dev_x)))
    dev_y = gen_one_hot(dev_y)

    train_x, train_y = read_data(train_file_path)

    # get vocab size
    with open(vocab_dic_path, 'rb') as fp:
        vocab_dict = pickle.load(fp)
        vocab_size = len(vocab_dict)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                max_sequence_lenth = max_sequence_lenth,
                vocab_size = vocab_size,
                embedding_size = embedding_dim,
                class_num = dev_y.shape[1],
                filter_sizes = list(map(int, filter_sizes.split(","))),
                filter_num = num_filters,
                l2_reg = l2_reg_lambda)


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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_TextCNN", p, timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.acc)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_Text_CNN", p)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)


            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.acc],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                data_size = len(x_batch)
                batch_num = 1000
                losses, accs = [], []
                for batch_num in range(data_size // batch_num + 1):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)

                    x = x_batch[start_index:end_index]
                    y = y_batch[start_index:end_index]

                    feed_dict = {
                      cnn.input_x: x,
                      cnn.input_y: y,
                      cnn.dropout_keep_prob: 1.0
                    }
                    loss, accuracy = sess.run(
                        [cnn.loss, cnn.acc],
                        feed_dict)

                    losses.append(loss)
                    accs.append(accuracy)

                val_loss = tf.reduce_mean(np.array(losses))
                val_acc = tf.reduce_mean(np.array(accs))

                val_loss_summary = tf.summary.scalar("loss", val_loss)
                val_acc_summary = tf.summary.scalar("accuracy", val_acc)
                dev_summary_op = tf.summary.merge([val_loss_summary, val_acc_summary])

                step, summaries, loss, acc = sess.run([global_step, dev_summary_op, val_loss, val_acc])
                time_str = datetime.datetime.now().isoformat()

                print("   {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                if writer:
                    writer.add_summary(summaries, step)
                val_res['acc'].append(acc)
                val_res['loss'].append(loss)

            batches = batch_iter(train_x, train_y, batch_size, num_epochs)

            for x_batch, y_batch in batches:
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\n-----------------\nEvaluation:\n-----------------")
                    val_step(dev_x, dev_y, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_TextCNN", p, timestamp))
    with open(os.path.join(out_dir, 'val_res'),'wb') as fp:
        pickle.dump(val_res, fp)

    print("finished !!!")


def main(argv=None):
    if data_option == 0:
        p = 'rough'
    else:
        p = 'rigour'
    train_data_path = os.path.join(data_dir, p, 'train')
    val_data_path = os.path.join(data_dir, p, 'val')
    vacab_data_path = '../data/trainSet/vacab.dic'
    train(train_data_path, val_data_path, vacab_data_path)


if __name__ == '__main__':
    tf.app.run()