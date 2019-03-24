#coding=utf-8

import datetime
import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf

from TextClassifier.data_helper import MAX_SENT_LEN_RIGOUR, MAX_SENT_NUM_RIGOUR, MAX_SENT_LEN_ROUGH, MAX_SENT_NUM_ROUGH
from TextClassifier.data_helper import stuff_doc, gen_one_hot, read_data
from classifier_tensorflow.HAN import HAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# # Data loading params
# tf.flags.DEFINE_string("data_dir", "../data/trainSet/Han", "data directory")
#
# # Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
# tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
# tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
#
# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches (default: 100)")
# tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate (default: 0.01)")
#
#
# FLAGS = tf.flags.FLAGS
#
# data_dir = FLAGS.data_dir
# embedding_dim = FLAGS.embedding_dim
# hidden_size = FLAGS.hidden_size
# grad_clip = FLAGS.grad_clip
# batch_size = FLAGS.batch_size
# num_epochs = FLAGS.num_epochs
# evaluate_every = FLAGS.evaluate_every
# checkpoint_every = FLAGS.checkpoint_every
# num_checkpoints = FLAGS.num_checkpoints
# learning_rate = FLAGS.learning_rate

data_dir = "../data/trainSet/Han"
embedding_dim = 200
hidden_size = 50
grad_clip = 5
batch_size = 256
num_epochs = 100
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
learning_rate = 0.01

data_option = 0

p = 'rough' if data_option == 0 else 'rigour'
max_sentence_length = MAX_SENT_LEN_ROUGH if data_option == 0 else MAX_SENT_LEN_RIGOUR
max_sentence_num = MAX_SENT_NUM_ROUGH if data_option == 0 else MAX_SENT_NUM_RIGOUR


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

            train_x = np.array(list(map(lambda doc: stuff_doc(doc, model_option=0, data_option=data_option), train_x_tmp)))
            train_y = gen_one_hot(train_y_tmp)

            yield train_x, train_y


def train(train_file_path, val_file_path, vocab_dic_path):

    # get all dev data
    dev_x, dev_y = read_data(val_file_path)
    dev_x = np.array(list(map(lambda doc: stuff_doc(doc, model_option=0, data_option=data_option), dev_x)))
    dev_y = gen_one_hot(dev_y)

    train_x, train_y = read_data(train_file_path)

    val_res = {
        'loss': [],
        'acc': [],
    }
    timestamp = None

    # get vocab size
    with open(vocab_dic_path, 'rb') as fp:
        vocab_dict = pickle.load(fp)
        vocab_size = len(vocab_dict)

    with tf.Session() as sess:
        han = HAN(
            vocab_size=vocab_size,
            num_classes=dev_y.shape[1],
            embedding_size=embedding_dim,
            hidden_size=hidden_size)


        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_Han", p, timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to {}\n".format(out_dir))

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        #RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(han.loss, tvars), grad_clip)
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

        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_Han", p)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.max_sentence_num: max_sentence_num,
                han.max_sentence_length: max_sentence_length,
            }
            _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, han.loss, han.acc],
                                                          feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return step

        def dev_step(x_batch, y_batch, writer=None):
            data_size = len(x_batch)
            batch_num = 1000
            losses, accs = [], []
            for batch_num in range(data_size // batch_num + 1):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                x = x_batch[start_index:end_index]
                y = y_batch[start_index:end_index]

                feed_dict = {
                    han.input_x: x,
                    han.input_y: y,
                    han.max_sentence_num: max_sentence_num,
                    han.max_sentence_length: max_sentence_length,
                }
                loss, accuracy = sess.run(
                    [han.loss, han.acc],
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
                dev_step(dev_x, dev_y, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_Han", p, timestamp))
    with open(os.path.join(out_dir, 'val_res'), 'wb') as fp:
        pickle.dump(val_res, fp)

    print("finished !!!")


def main(argv=None):
    if data_option == 0:
        p = 'rough'
    else:
        p = 'rigour'
    train_data_path = os.path.join(data_dir, p, 'train')
    val_data_path = os.path.join(data_dir, p, 'val')
    vacab_data_path = '../data/trainSet/vocab.dic'
    train(train_data_path, val_data_path, vacab_data_path)



if __name__ == '__main__':
    tf.app.run()


