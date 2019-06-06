import pickle
import numpy as np
import math
import keras


MAX_LEN = 200


def load_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        self.indexes = np.arange(len(self.data['X1']))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_data_x1 = [self.data['X1'][k] for k in batch_indexs]
        batch_data_x2 = [self.data['X2'][k] for k in batch_indexs]
        batch_data_y = [self.data['Y'][k] for k in batch_indexs]

        # 生成数据
        X1, X2, y = self.data_generation(batch_data_x1, batch_data_x2, batch_data_y)
        return [X1,X2], y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __stuff_doc(self, doc):
        doc = [int(x) for x in doc.split(' ')]
        doc = doc * math.ceil(MAX_LEN/len(doc))
        return doc[:MAX_LEN]

    def data_generation(self, batch_data_x1, batch_data_x2, batch_data_y):
        train_x1 = np.array(list(map(lambda doc: self.__stuff_doc(doc), batch_data_x1)))
        train_x2 = np.array(list(map(lambda doc: self.__stuff_doc(doc), batch_data_x1)))
        train_y = np.array(batch_data_y)

        return train_x1, train_x2, train_y