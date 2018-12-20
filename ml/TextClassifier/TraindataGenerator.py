# coding:utf-8

from abc import abstractmethod, ABCMeta
import pandas as pd
import pickle
import os
from collections import defaultdict




class TfrecordGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, src_path):
        self.name = name

        self.option = None
        self.code2label = None
        self.vocab = None
        self.text_col = None

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


    def get_dict(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as fp:
                self.vocab = pickle.load(fp)
        else:
            all_text = self.df['rigour_token'].tolist()
            tmp = defaultdict(lambda : 0)

            for text in all_text:
                for word in text.replace('。', ' ').split(' '):
                    tmp[word] += 1

            tmp = [(k, v) for k, v in tmp.items()]
            tmp.sort(key=lambda x:x[1], reverse=True)

            tmp = filter(lambda x: x[1] > 10, tmp)
            tmp = [x[0] for x in tmp]

            self.vocab = dict()
            for index, word in enumerate(tmp):
                self.vocab[word] = index

            print('writing dic...')
            with open(path, 'wb') as fp:
                pickle.dump(self.vocab, fp)

    @abstractmethod
    def process_text(self, doc):
        pass

    def process_data(self):

        print('train data')
        # 训练集
        self.x_train = self.train_data[self.text_col].apply(self.process_text).tolist()
        train_y = self.train_data['cls'].tolist()

        self.code2label = {}
        for index, item in enumerate(list(set(train_y))):
            self.code2label[item] = index

        self.y_train = list(map(lambda x: self.code2label[x], train_y))

        print('val data')
        # 验证集
        self.x_val = self.val_data[self.text_col].apply(self.process_text).tolist()
        self.y_val = list(map(lambda x: self.code2label[x], self.val_data['cls'].tolist()))

        print('test data')
        # Test set
        self.x_test = self.test_data[self.text_col].apply(self.process_text).tolist()
        self.y_test = list(map(lambda x: self.code2label[x], self.test_data['cls'].tolist()))


    def out(self, path):
        if self.option == 0:
            tmp = 'rough'
        else:
            tmp = 'rigour'

        dir = os.path.join(path, self.name, tmp)
        if not os.path.exists(dir):
            os.makedirs(dir)

        print('processing...')
        self.process_data()

        print('out cls_map...')
        with open(os.path.join(dir, 'cls_label.dic'), 'wb') as fp:
            pickle.dump(self.code2label, fp)

        print('out train data...')
        with open(os.path.join(dir, 'train'), 'wb') as fp:
            pickle.dump((self.x_train, self.y_train), fp)

        print('out val data...')
        with open(os.path.join(dir, 'val'), 'wb') as fp:
            pickle.dump((self.x_val, self.y_val), fp)

        print('out test data...')
        with open(os.path.join(dir, 'test'), 'wb') as fp:
            pickle.dump((self.x_test, self.y_test), fp)


class HanTfrecordGenerator(TfrecordGenerator):
    def __init__(self, src_path):
        TfrecordGenerator.__init__(self, 'Han', src_path)

        print('Han start...')

    def preprocess_data(self, option=0):
        if option == 0:
            self.option = 0
            self.text_col = 'rough_token'
        elif option == 1:
            self.option = 1
            self.text_col = 'rigour_token'
        else:
            raise ('option par error! (must be 0 or 1)')

        self.data = self.df.loc[:, [self.text_col, 'cls', 'train_val_test']]

        TfrecordGenerator.preprocess_data(self)


    def process_text(self, doc):
        res = []
        for sent in doc.split('。'):
            tmp = map(lambda x: self.vocab[x] if x in self.vocab else None, sent.split(' '))
            tmp = list(filter(lambda x: x is not None, tmp))
            if len(tmp) > 0:
                res.append(tmp)
        return res



class TextCnnTfrecordGenerator(TfrecordGenerator):
    def __init__(self, src_path):

        TfrecordGenerator.__init__(self, 'TextCnn', src_path)

        print('TextCnn start...')

    def preprocess_data(self, option=0):
        if option == 0:
            self.option = 0
            self.text_col = 'rough_token'
        elif option == 1:
            self.option = 1
            self.text_col = 'rigour_token'
        else:
            raise ('option par error! (must be 0 or 1)')

        self.data = self.df.loc[:, [self.text_col, 'cls', 'train_val_test']]

        TfrecordGenerator.preprocess_data(self)


    def process_text(self, doc):
        tmp = list(map(lambda x: self.vocab[x] if x in self.vocab else None, doc.replace('。', ' ').split(' ')))
        return tmp



if __name__ == '__main__':
    data_path = '../data/trainSet/train_info_5w.csv'

    htg = HanTfrecordGenerator(data_path)
    print('rough...')
    htg.preprocess_data(option=0)
    htg.get_dict('../data/trainSet/vacb.dic')
    htg.out(path='../data/trainSet')
    print('rigour...')
    htg.preprocess_data(option=1)
    htg.get_dict('../data/trainSet/vacb.dic')
    htg.out(path='../data/trainSet')

    tctg = TextCnnTfrecordGenerator(data_path)
    print('rough...')
    tctg.preprocess_data(option=0)
    tctg.get_dict('../data/trainSet/vacb.dic')
    tctg.out(path='../data/trainSet')
    print('rigour...')
    tctg.preprocess_data(option=1)
    tctg.get_dict('../data/trainSet/vacb.dic')
    tctg.out(path='../data/trainSet')

    print('finished!')
