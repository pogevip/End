import pickle
import numpy as np
import math
import keras


MAX_LEN = 200
CLASS_NUM = 10


def load_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def load_cls_weight(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    weight_dic = [[item[0], item[1]] for _, item in dic.items()]
    weight_dic.sort(key=lambda x:x[0])
    weight = np.array([item[1] for item in weight_dic])
    return weight

def load_cls_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    weight_dic = [[item[0], cls] for cls, item in dic.items()]
    weight_dic.sort(key=lambda x:x[0])
    cls_list = [item[1] for item in weight_dic]
    return cls_list


class RandomSampleDataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        self.indexes = np.arange(len(self.data['X']))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[int(index*self.batch_size) : int((index+1)*self.batch_size)]
        # 根据索引获取datas集合中的数据
        batch_data_x = [self.data['X'][k] for k in batch_indexs]
        batch_data_y = [self.data['Y'][k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_data_x, batch_data_y)
        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __stuff_doc(self, doc):
        doc = doc * math.ceil(MAX_LEN/len(doc))
        return doc[:MAX_LEN]

    def __gen_one_hot(self, label):
        label = np.array(label)
        return (np.arange(CLASS_NUM)==label[:,None]).astype(np.int32)

    def data_generation(self, batch_data_x, batch_data_y):
        train_x = np.array(list(map(lambda doc: self.__stuff_doc(doc), batch_data_x)))
        train_y = self.__gen_one_hot(batch_data_y)

        return train_x, train_y

class StratifiedSampleDataGenerator(keras.utils.Sequence):
    #对于数据分层采样，每个batch都按比例分布
    def __init__(self, data, batch_prop, shuffle=True):
        self.data = data
        self.batch_prop = batch_prop
        self.indexes = {cls :np.arange(len(d['X'])) for cls, d in data.items()}
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(1 / self.batch_prop)

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_data_x = []
        batch_data_y = []
        for cls, d in self.data.items():
            batch_size = len(d['X']) * self.batch_prop
            batch_indexs = self.indexes[cls][int(index * batch_size): int((index + 1) * batch_size)]

            batch_data_x.extend([d['X'][k] for k in batch_indexs])
            batch_data_y.extend([d['Y'][k] for k in batch_indexs])

        # 生成数据
        X, y = self.data_generation(batch_data_x, batch_data_y)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            for _, i in self.indexes.items():
                np.random.shuffle(i)

    def __stuff_doc(self, doc):
        doc = doc * math.ceil(MAX_LEN/len(doc))
        return doc[:MAX_LEN]

    def __gen_one_hot(self, label):
        label = np.array(label)
        return (np.arange(CLASS_NUM)==label[:,None]).astype(np.int32)

    def data_generation(self, batch_data_x, batch_data_y):
        train_x = np.array(list(map(lambda doc: self.__stuff_doc(doc), batch_data_x)))
        train_y = self.__gen_one_hot(batch_data_y)

        return train_x, train_y


# def DataGenerator(data, batch, is_train_data, shuffle=True):
#     def stuff_doc(doc):
#         doc = doc * math.ceil(MAX_LEN/len(doc))
#         return doc[:MAX_LEN]
#
#     def gen_one_hot(label):
#         label = np.array(label)
#         return (np.arange(CLASS_NUM)==label[:,None]).astype(np.int32)
#
#     if is_train_data:
#         indexes = {cls :np.arange(len(d['X'])) for cls, d in data.items()}
#         num_batches_per_epoch = int(1 / batch)
#
#         while True:
#             if shuffle:
#                 for _, i in indexes.items():
#                     np.random.shuffle(i)
#
#             for batch_index in range(int(num_batches_per_epoch)):
#                 batch_data_x = []
#                 batch_data_y = []
#                 for cls, d in data.items():
#                     batch_size = len(d['X'])*batch
#                     batch_indexs = indexes[cls][batch_index * batch_size: (batch_index + 1) * batch_size]
#
#                     batch_data_x.extend([d['X'][k] for k in batch_indexs])
#                     batch_data_y.extend([d['Y'][k] for k in batch_indexs])
#
#                 batch_data_x = np.array(list(map(lambda doc: stuff_doc(doc), batch_data_x)))
#                 batch_data_y = gen_one_hot(batch_data_y)
#
#                 yield batch_data_x, batch_data_y
#     else:
#         indexes = np.arange(len(data['X']))
#         batch_size = batch
#         num_batches_per_epoch = math.ceil(len(data) / batch_size)
#
#         while True:
#             if shuffle:
#                 np.random.shuffle(indexes)
#
#             for batch_index in range(int(num_batches_per_epoch)):
#                 batch_indexs = indexes[batch_index * batch_size: (batch_index + 1) * batch_size]
#
#                 batch_data_x = [data['X'][k] for k in batch_indexs]
#                 batch_data_y = [data['Y'][k] for k in batch_indexs]
#
#                 batch_data_x = np.array(list(map(lambda doc: stuff_doc(doc), batch_data_x)))
#                 batch_data_y = gen_one_hot(batch_data_y)
#
#                 yield batch_data_x, batch_data_y

