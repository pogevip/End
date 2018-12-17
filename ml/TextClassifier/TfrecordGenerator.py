# coding:utf-8

from abc import abstractmethod, ABCMeta
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

from PrepareData.gen_train_data import MAX_SENT_NUM_ROUGH, MAX_SENT_LEN_ROUGH, MAX_DOC_LEN_ROUGH, \
    MAX_SENT_NUM_RIGOUR, MAX_SENT_LEN_RIGOUR, MAX_DOC_LEN_RIGOUR


class TfrecordGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, src_path):
        self.name = name

        self.option = None
        self.code2label = None
        self.vocab_processor = None

        self.data = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        print('load data...')
        self.df = pd.read_csv(src_path)

    def preprocess_data(self):

        self.train_data = self.data[self.data['train_val_test'] == 1]
        self.val_data = self.data[self.data['train_val_test'] == 2]
        self.test_data = self.data[self.data['train_val_test'] == 3]

        self.train_data.dropna(how='any', inplace=True)
        self.val_data.dropna(how='any', inplace=True)
        self.test_data.dropna(how='any', inplace=True)

        print('load finished.')
        print('train/val/test = {}/{}/{}'.format(len(self.train_data), len(self.val_data), len(self.test_data)))

        print('processing...')

    @abstractmethod
    def process_data(self):
        pass


    def to_tfrecord(self, features, labels, out_path):
        writer = tf.python_io.TFRecordWriter(out_path)

        for feature, label in zip(features, labels):
            item = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                    'input_raw': tf.train.Feature(int64_list=tf.train.Int64List(value=feature.tolist()))
                }))
            writer.write(item.SerializeToString())

        writer.close()

    def out(self, path):
        if self.option == 0:
            tmp = 'rough'
        else:
            tmp = 'rigour'

        dir = os.path.join(path, self.name, tmp)
        if not os.path.exists(dir):
            os.mkdir(dir)

        self.process_data()

        print('out cls_map...')
        with open(os.path.join(dir, 'cls_label.dic'), 'wb') as fp:
            pickle.dump(self.code2label, fp)

        print('out voc_dic...')
        with open(os.path.join(dir, 'vocab.dic'), 'wb') as fp:
            pickle.dump(self.vocab_processor.vocabulary_._mapping, fp)

        print('out train data...')
        self.to_tfrecord(self.x_train, self.y_train, os.path.join(dir, 'train.tfrecords'))
        print('out val data...')
        self.to_tfrecord(self.x_val, self.y_val, os.path.join(dir, 'val.tfrecords'))
        print('out test data...')
        self.to_tfrecord(self.x_test, self.y_test, os.path.join(dir, 'test.tfrecords'))


class HanTfrecordGenerator(TfrecordGenerator):
    def __init__(self, src_path):
        TfrecordGenerator.__init__(self, 'Han', src_path)

        print('Han start...')

    def preprocess_data(self, option=0):
        if option == 0:
            self.option = 0
            self.text_col = 'rough_token'
            self.max_sentence_length = MAX_SENT_LEN_ROUGH
            self.max_document_length = MAX_SENT_NUM_ROUGH
        elif option == 1:
            self.option = 1
            self.text_col = 'rigour_token'
            self.max_sentence_length = MAX_SENT_LEN_RIGOUR
            self.max_document_length = MAX_SENT_NUM_RIGOUR
        else:
            raise ('option par error! (must be 0 or 1)')

        self.data = self.df.loc[:, [self.text_col, 'cls', 'train_val_test']]

        TfrecordGenerator.preprocess_data(self)

    def process_data(self):
        def helper(doc):
            sents = filter(lambda x: len(x.strip())>0, doc.split('。'))
            sents = map(lambda sent: list(filter(lambda x: len(x.strip())>0, sent.split(' '))), sents)
            return list(sents)

        train_x_text = map(lambda x: helper(x), self.train_data[self.text_col].tolist())

        train_y = self.train_data['cls'].tolist()

        self.code2label = {}
        for index, item in enumerate(list(set(train_y))):
            self.code2label[item] = index

        self.y_train = np.array(map(lambda x: self.code2label[x], train_y))

        # 验证集
        val_x_text = map(lambda x: helper(x), self.val_data[self.text_col].tolist())

        self.y_val = np.array(map(lambda x: self.code2label[x], self.val_data['cls'].tolist()))

        # Build vocabulary
        text_tmp = []
        text_tmp.extend(train_x_text)
        text_tmp.extend(val_x_text)

        tmp = np.array(text_tmp).reshape(-1)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_sentence_length)  # 将每个文档都填充成最大长度，0填充
        self.vocab_processor.fit(tmp)

        x_train = []
        for doc in train_x_text:
            each_doc = np.array(list(self.vocab_processor.fit_transform(doc)))
            x_train.append(np.pad(each_doc, ((0, self.max_document_length - len(each_doc)), (0, 0)), 'constant'))
        self.x_train = np.array(x_train)

        x_val = []
        for doc in val_x_text:
            each_doc = np.array(list(self.vocab_processor.fit_transform(doc)))
            x_val.append(np.pad(each_doc, ((0, self.max_document_length - len(each_doc)), (0, 0)), 'constant'))
        self.x_val = np.array(x_val)

        # Test set
        test_x_text = map(lambda x: helper(x), self.test_data[self.text_col].tolist())
        x_test = []
        for doc in test_x_text:
            each_doc = np.array(list(self.vocab_processor.fit_transform(doc)))
            x_test.append(np.pad(each_doc, ((0, self.max_document_length - len(each_doc)), (0, 0)), 'constant'))
        self.x_test = np.array(x_test)

        self.y_test = np.array(map(lambda x: self.code2label[x], self.test_data['cls'].tolist()))

        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))

    def to_tfrecord(self, features, labels, out_path):
        features = np.reshape(features, [-1, self.max_document_length * self.max_sentence_length])
        TfrecordGenerator.to_tfrecord(self, features, labels, out_path)


class TextCnnTfrecordGenerator(TfrecordGenerator):
    def __init__(self, src_path):

        TfrecordGenerator.__init__(self, 'TextCnn', src_path)

        print('TextCnn start...')

    def preprocess_data(self, option=0):
        if option == 0:
            self.option = 0
            self.text_col = 'rough_token'
            self.max_document_length = MAX_DOC_LEN_ROUGH
        elif option == 1:
            self.option = 1
            self.text_col = 'rigour_token'
            self.max_document_length = MAX_DOC_LEN_RIGOUR
        else:
            raise ('option par error! (must be 0 or 1)')

        self.data = self.df.loc[:, [self.text_col, 'cls', 'train_val_test']]

        TfrecordGenerator.preprocess_data(self)

    def process_data(self):
        def helper(doc):
            doc = doc.replace('。', ' ')
            doc = filter(lambda x: len(x.strip())>0, doc.split(' '))
            return list(doc)

        train_x_text = map(lambda x: helper(x), self.train_data[self.text_col].tolist())

        train_y = self.train_data['cls'].tolist()

        self.code2label = {}
        for index, item in enumerate(list(set(train_y))):
            self.code2label[item] = index

        self.y_train = np.array(map(lambda x: self.code2label[x], train_y))

        # 验证集
        val_x_text = map(lambda x: helper(x), self.val_data[self.text_col].tolist())

        self.y_val = np.array(map(lambda x: self.code2label[x], self.val_data['cls'].tolist()))

        # Build vocabulary
        text_tmp = []
        text_tmp.extend(train_x_text)
        text_tmp.extend(val_x_text)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length)  # 将每个文档都填充成最大长度，0填充
        self.vocab_processor.fit(text_tmp)

        self.x_train = np.array(list(self.vocab_processor.fit_transform(train_x_text)))

        self.x_val = np.array(list(self.vocab_processor.fit_transform(val_x_text)))

        # Test set
        test_x_text = map(lambda x: helper(x), self.test_data[self.text_col].tolist())
        self.x_test = np.array(list(self.vocab_processor.fit_transform(test_x_text)))

        self.y_test = np.array(map(lambda x: self.code2label[x], self.test_data['cls'].tolist()))

        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))

    def to_tfrecord(self, features, labels, out_path):
        TfrecordGenerator.to_tfrecord(self, features, labels, out_path)


if __name__ == '__main__':
    data_path = 'train_set/train_info.csv'

    htg = HanTfrecordGenerator(data_path)
    htg.preprocess_data()
    htg.out(path='../data/train_data')

    tctg = TextCnnTfrecordGenerator(data_path)
    tctg.preprocess_data()
    tctg.out(path='../data/train_data')
