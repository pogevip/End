#coding:utf-8

from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


TRAIN_DATA_PATH = 'data/train_data/HAN_train.tfrecords'
DEV_DATA_PATH = 'data/train_data/HAN_dev.tfrecords'
TEST_DATA_PATH = 'data/train_data/HAN_test.tfrecords'
VOCAB_DICT_PATH = 'data/train_data/HAN_vocab.dic'


data_size_config = dict()


def load_stop_words(path = 'data/stopWords.txt'):
    stw = []
    with open(path, 'r') as fp:
        for line in fp:
            stw.append(line.strip())
    return stw

stop_words = load_stop_words()
stop_flags = ['']


def preprocess_str(string, with_flag = True):
    res = []
    for sentence in string.split('。-x'):
        sentence = sentence.strip()
        if sentence != '':
            tmp = []
            for wf in sentence.split(' '):
                word, flag = wf.split('-')
                if word not in stop_words:
                    if with_flag and flag not in stop_flags:
                        tmp.append(word)
                    else:
                        tmp.append(word)
            if len(tmp)>0:
                res.append(tmp)
    return res


def read_dataset(data_path, dev_sample_percentage=0.1):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    data1 = df[df['tag'] == True]
    data2 = df[df['tag'] == False]

    x_text = [preprocess_str(x) for x in data1['text'].tolist()]

    y = data1['label'].tolist()

    code2label = {}
    for index ,item in enumerate(list(set(y))):
        code2label[item] = index
    with open('data/train_data/HAN_cls_label.dic', 'wb') as fp:
        pickle.dump(code2label, fp)

    y = np.array([code2label[_] for _ in y])
    y = (np.arange(len(code2label)) == y[:,None]).astype(np.int)

    # Build vocabulary
    max_document_length = max([len(x) for x in x_text]) + 3
    max_sentence_length = np.max([[len(x.split(' ')) for x in doc] for doc in x_text])

    data_size_config['MAX_DOC_LEN'] = max_document_length
    data_size_config['MAX_SEN_LEN'] = max_sentence_length

    with open('data/train_data/HAN_data_size.config', 'wb') as fp:
        pickle.dump(data_size_config, fp)

    tmp = np.array(x_text).reshape(-1)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length) #将每个文档都填充成最大长度，0填充
    vocab_processor.fit(tmp)

    tmp1 = []
    for doc in x_text:
        each_doc = np.array(list(vocab_processor.fit_transform(doc)))
        tmp1.append(np.pad(each_doc, ((0, max_document_length - len(each_doc)), (0, 0)), 'constant'))
    x = np.array(tmp1)


    # Test set
    x_test = [preprocess_str(x) for x in data2['text'].tolist()]
    tmp2 = []
    for doc in x_test:
        each_doc = np.array(list(vocab_processor.fit_transform(doc)))
        tmp2.append(np.pad(each_doc, ((0, max_document_length - len(each_doc)), (0, 0)), 'constant'))
    x_test = np.array(tmp2)

    y_test = np.array([code2label[_] for _ in data2['label'].tolist()])
    y_test = (np.arange(len(code2label)) == y_test[:, None]).astype(np.int)


    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev , x_test, y_test


def out_tfrecord(features, labels, out_path):
    writer = tf.python_io.TFRecordWriter(out_path)

    for feature, label in zip(features, labels):
        feature = np.reshape(feature, [data_size_config['MAX_DOC_LEN'] * data_size_config['MAX_SEN_LEN']])
        item = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'input_raw': tf.train.Feature(float_list=tf.train.Int64List(value=feature.tolist()))
            }))
        writer.write(item.SerializeToString())

    writer.close()


def gen_vocab_map(vocab_processor, path = VOCAB_DICT_PATH):
    with open(path, 'wb') as fp:
        pickle.dump(vocab_processor.vocabulary_._mapping, fp)


if __name__ == '__main__':
    path = ''
    x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test = read_dataset(path)

    out_tfrecord(x_train, y_train, out_path=TRAIN_DATA_PATH)
    out_tfrecord(x_dev, y_dev, out_path=DEV_DATA_PATH)
    out_tfrecord(x_test, y_test, out_path=TEST_DATA_PATH)

    gen_vocab_map(vocab_processor)