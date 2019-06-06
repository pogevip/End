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
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("word_text_cnn_train.txt", mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



data_dir = "../data/trainSet/WordTextCnn"
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

max_sequence_lenth = 2500


class DataReader():
    def __init__(self, path, batch_size=0, num_epochs=num_epochs):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        self.x = data[0]
        self.y = data[1]
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_dict_len(self, path):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        return len(data)

    def read(self):
        def stuff_doc(doc, max_length=2500):
            if len(doc) > max_length:
                start_index = (len(doc) - max_length - 1) // 2 + 1
                doc = doc[start_index: start_index + max_length]
            else:
                doc.extend([0] * (max_length - len(doc)))
            return doc

        def gen_one_hot(label):
            label = np.array(label)
            return (np.arange(10) == label[:, None]).astype(np.int32)

        if self.batch_size == 0:
            x = np.array(list(map(lambda doc: stuff_doc(doc), self.x)))
            y = gen_one_hot(self.y)
            return x, y
        else:
            data_size = len(self.x)
            num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1

            for epochs in range(self.num_epochs):
                print('epoch:', epochs)
                shuffle_indices = list(range(data_size))
                random.shuffle(shuffle_indices)

                shuffled_x = [self.x[i] for i in shuffle_indices]
                shuffled_y = [self.y[i] for i in shuffle_indices]


                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * self.batch_size
                    end_index = min((batch_num + 1) * self.batch_size, data_size)

                    x_tmp = shuffled_x[start_index:end_index]
                    y_tmp = shuffled_y[start_index:end_index]

                    x = np.array(list(map(lambda doc: stuff_doc(doc), x_tmp)))
                    y = gen_one_hot(y_tmp)

                    yield x, y



def train(train_file_path, val_file_path, vocab_dic_path):
    # Training
    # ==================================================

    # get all dev data

    val_res = {
        'loss' : [],
        'acc' : [],
    }

    val_reader = DataReader(val_file_path)
    dev_x, dev_y = val_reader.read()

    train_reader = DataReader(train_file_path, batch_size=batch_size, num_epochs=num_epochs)

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
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_WordTextCNN", timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logger.info("Writing to {}\n".format(out_dir))

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
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_Word_Text_CNN")
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
                logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
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

                logger.info("   {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                if writer:
                    writer.add_summary(summaries, step)
                val_res['acc'].append(acc)
                val_res['loss'].append(loss)


            for x_batch, y_batch in train_reader.read():
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    logger.info("\n-----------------\nEvaluation:\n-----------------")
                    val_step(dev_x, dev_y, writer=dev_summary_writer)
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {}\n".format(path))

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_WordTextCNN", timestamp))
    with open(os.path.join(out_dir, 'val_res'),'wb') as fp:
        pickle.dump(val_res, fp)

    print("finished !!!")


def main(argv=None):
    train_data_path = os.path.join(data_dir, 'train')
    val_data_path = os.path.join(data_dir, 'val')
    vacab_data_path = '../data/trainSet/vacab.dic'
    train(train_data_path, val_data_path, vacab_data_path)


if __name__ == '__main__':
    tf.app.run()