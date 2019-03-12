import pandas as pd
import numpy as np
import pickle
from collections import Counter

def load_path(path):
    data = pd.read_csv(path)
    return data['cls'].tolist()

class cls_dic_weight_generator():
    def __init__(self, data, out_path):
        self.data = data
        self.path = out_path

    def count_cls(self):
        cls_count = list(Counter(self.data).items())
        return cls_count

    def __softmax(self, arr):
        arr = np.array(arr)
        exp_x = np.exp(arr)
        return exp_x / np.sum(exp_x)

    def gen_cls_dict(self, cls_count):
        cls_count.sort(key=lambda x:x[1])

        clss = [item[0] for item in cls_count]
        counts = [item[1] for item in cls_count]

        weights = list(self.__softmax(counts))
        weights.reverse()

        dic = {}
        for i in range(len(clss)):
            dic[clss[i]] = (i, weights[i])

        with open(self.path, 'wb') as fp:
            pickle.dump(dic, fp)

    def run(self):
        cls_count = self.count_cls()
        self.gen_cls_dict(cls_count)
        print('finished!')


if __name__ == '__main__':
    from_path = ''
    out_path = 'data/trainSet/classifier/cls_dic_weight.dic'
    data = load_path(from_path)

    adw_gen = cls_dic_weight_generator(data, out_path)
    adw_gen.run()