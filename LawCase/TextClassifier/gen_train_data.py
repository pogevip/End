# coding:utf-8

import pandas as pd
import random
from pymongo import MongoClient
import math

conn = MongoClient('172.19.241.248', 20000)
col = conn['wangxiao']['alldata_final']

MAX_DOC_LEN = 400

stop_flags = ['b', 'c', 'e', 'g', 'h', 'k', 'l', 'o', 's', 'u', 'w', 'x', 'y', 'z', 'un', 'nr', 'ns',
              'f', 'i', 'm', 'p', 'q', 'r', 'tg', 't']


def gen_data_to_csv(path):
    res = []
    index = 0
    demo = col.find(no_cursor_timeout=True)
    for item in demo:
        try:
            token = item['token']
            token = [x.split('/') for x in token.split(' ')]
            token = filter(lambda x: x[1] not in stop_flags, token)
            token = [x[0] for x in token]
        except:
            continue

        if len(token) >= 400:
            continue
        else:
            res.append([item['fullTextId'], ' '.join(token), item['cls']])

        if index % 10000 == 0:
            print(index)
        index += 1
    demo.close()

    res = pd.DataFrame(res, columns=['id', 'token', 'cls'])
    res.dropna(how='any', inplace=True)
    res['len'] = res['token'].apply(lambda x: len(x.split(' ')))
    res.to_csv(path, index=0)
    print('finished!')


class TrainData():
    def __init__(self, src_path):
        self.data = pd.read_csv(src_path)
        self.data['is_gen'] = 0
        self.res = pd.DataFrame([], columns=['id', 'token', 'cls', 'is_gen', 'len'])

    def sample(self, min_num, max_num):
        # 对少数数据进行采样

        def random_delete_word(word, threshold=0.15):
            if random.random() >= threshold:
                return word

        def process_token(doc):
            doc = doc.split(' ')
            doc = doc if len(doc) <= 200 else doc[:200]
            doc = list(filter(lambda x: x, map(random_delete_word, doc)))
            return ' '.join(doc)

        for cls, group in self.data.groupby('cls'):
            print(cls, len(group))
            if len(group) < min_num:
                tmp = group[group['len'] >= 20]
                buffer = []
                multiple = int(math.ceil(min_num / len(tmp)))
                for _, row in tmp.iterrows():
                    buffer.append([row['id'], row['token'], row['cls'], row['is_gen'], row['len']])
                    for i in range(multiple):
                        token = process_token(row['token'])
                        buffer.append([row['id'], token, row['cls'], 1, len(token.split(' '))])

                df = pd.DataFrame(buffer, columns=['id', 'token', 'cls', 'is_gen', 'len'])
                df = df.sample(min_num)
                self.res = pd.concat([self.res, df], ignore_index=True)
            elif len(group) > max_num:
                if len(group[(group['len'] <= 200) & (group['len'] >= 20)]) > max_num:
                    tmp = group[(group['len'] <= 200) & (group['len'] >= 20)].sample(max_num)
                else:
                    tmp = group[(group['len'] <= 200) & (group['len'] >= 20)]
#                     tmp = tmp.sample(len(tmp) // 10000 * 10000)
                self.res = pd.concat([self.res, tmp], ignore_index=True)
            else:
                tmp = group[group['len'] >= 20]
#                 tmp = tmp.sample(len(tmp) // 10000 * 10000)
                self.res = pd.concat([self.res, tmp], ignore_index=True)

        self.res['train_val_test'] = 1

    def split_train_val_test_set(self):
        # 划分测试集
        for _, group in self.res.groupby('cls'):
#             num = (math.ceil((len(group)//10000+0.1)*0.2)-1)*10000
            test_index = group.sample(frac=0.2).index
            self.res.loc[test_index, 'train_val_test'] = 3

        # 划分验证集
        test_df = self.res[self.res['train_val_test'] == 3]
        for _, group in test_df.groupby('cls'):
            val_index = group.sample(frac=0.6).index
            self.res.loc[val_index, 'train_val_test'] = 2

    def output_to_csv(self, path):
        print('all info length: {}'.format(len(self.res)))
        self.res.to_csv(path, index=0)

    def run(self, min_num=20000, max_num=50000):
        self.sample(min_num, max_num)
        self.split_train_val_test_set()


if __name__ == '__main__':
    gen_data_to_csv('../data/data.csv')

    td = TrainData('../data/data.csv')

    print('process...')
    td.run()

    print('out...')
    td.output_to_csv('../data/trainSet/train_info_5w.csv')

    print('finished!')