#coding:utf-8

import pandas as pd
import random


MAX_SENT_NUM_ROUGH = 20
MAX_SENT_LEN_ROUGH = 150
MAX_DOC_LEN_ROUGH = 500

MAX_SENT_NUM_RIGOUR = 20
MAX_SENT_LEN_RIGOUR = 100
MAX_DOC_LEN_RIGOUR = 400


class TrainData():
    def __init__(self, src_path):
        self.all_info = pd.read_csv(src_path)
        self.all_info.dropna(how='any', inplace=True)

        self.all_info = self.all_info[(self.all_info['sent_num_rough'] <= MAX_SENT_NUM_ROUGH) &
                                      (self.all_info['max_sentlen_rough'] <= MAX_SENT_LEN_ROUGH) &
                                      (self.all_info['doc_len_rough'] <= MAX_DOC_LEN_ROUGH) &
                                      (self.all_info['sent_num_rigour'] <= MAX_SENT_NUM_RIGOUR) &
                                      (self.all_info['max_sentlen_rigour'] <= MAX_SENT_LEN_RIGOUR) &
                                      (self.all_info['doc_len_rigour'] <= MAX_DOC_LEN_RIGOUR)]

        self.all_info['is_gen'] = 0
        self.res = pd.DataFrame([], columns = ['rough_token', 'rigour_token',
                                                'sent_num_rough', 'max_sentlen_rough', 'doc_len_rough',
                                                'sent_num_rigour', 'max_sentlen_rigour', 'doc_len_rigour',
                                                'cls', 'is_gen'])


    def sample(self, base_line = 50000):
        #对少数数据进行采样

        def random_delete_word(word, threshold=0.15):
            if random.random() >= threshold:
                return word

        def process_token(doc):
            res = []
            for sen in doc.split('。'):
                sen = filter(lambda x: x, map(random_delete_word, sen.split(' ')))
                res.append(' '.join(sen))
            return '。'.join(res)

        def com_len(doc):
            sents = doc.split('。')
            sent_num = len(sents)
            each_sent_len = list(map(lambda x: len(x.split(' ')), sents))
            max_sentlen = max(each_sent_len)
            doc_len = sum(each_sent_len)
            return sent_num, max_sentlen, doc_len

        for cls, group in self.all_info.groupby('cls'):
            print(cls, len(group))
            if len(group)  < base_line:
                buffer = []
                multiple = int(base_line/len(group)+0.8)
                for _, row in group.iterrows():
                    for i in range(multiple):

                        rough_token = process_token(row['rough_token'])
                        rigour_token = process_token(row['rigour_token'])

                        sent_num1, max_sentlen1, doc_len1 = com_len(rough_token)
                        sent_num2, max_sentlen2, doc_len2 = com_len(rigour_token)

                        buffer.append([rough_token, rigour_token,
                                       sent_num1, max_sentlen1, doc_len1,
                                       sent_num2, max_sentlen2, doc_len2,
                                       row['cls'], 1])

                df = pd.DataFrame(buffer, columns = ['rough_token', 'rigour_token',
                                                     'sent_num_rough', 'max_sentlen_rough', 'doc_len_rough',
                                                     'sent_num_rigour', 'max_sentlen_rigour', 'doc_len_rigour',
                                                     'cls', 'is_gen'])

                self.res = pd.concat([self.res, df], sort=True, ignore_index=True)
            elif len(group) > base_line * 5:
                tmp = group.sample(base_line * 5)
                self.res = pd.concat([self.res, tmp], sort=True, ignore_index=True)
            else:
                self.res = pd.concat([self.res, group], sort=True, ignore_index=True)



        self.res = self.res.loc[:, ['rough_token', 'rigour_token',
                                              'sent_num_rough', 'max_sentlen_rough', 'doc_len_rough',
                                              'sent_num_rigour', 'max_sentlen_rigour', 'doc_len_rigour',
                                              'cls', 'is_gen']]
        self.res['train_val_test'] = 1


    def split_train_val_test_set(self, frac=0.2, max_num=30000):
        #划分测试集
        for _, group in self.res.groupby('cls'):
            if len(group) * frac <= max_num:
                test_index = group.sample(frac=frac).index
                self.res.loc[test_index, 'train_val_test'] = 3
            else:
                test_index = group.sample(max_num).index
                self.res.loc[test_index, 'train_val_test'] = 3

        #划分验证集
        train_df = self.res[self.res['train_val_test'] == 1]
        for _, group in train_df.groupby('cls'):
            if len(group) * frac <= max_num:
                val_index = group.sample(frac=frac).index
                self.res.loc[val_index, 'train_val_test'] = 2
            else:
                val_index = group.sample(max_num).index
                self.res.loc[val_index, 'train_val_test'] = 2


    def output_to_csv(self, path):
        print('all info length: {}'.format(len(self.res)))
        self.res.to_csv(path, index=0)


if __name__ == '__main__':
    td = TrainData('../data/all_info.csv')

    print('up sample...')
    td.sample()

    print('spliting...')
    td.split_train_val_test_set()

    print('out...')
    td.output_to_csv('../data/trainSet/train_info_5w.csv')

    print('finished!')