from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pymongo import MongoClient, ASCENDING
import os
import pickle
from collections import Counter

conn = MongoClient('172.19.241.248', 20000)

db = conn['wangxiao']


def trans_data_to_file(path='data/data.csv', out_dir='data/trainSet/rank/tfidf/'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = pd.read_csv(path)
    for cls, group in data.groupby('cls'):
        print(cls)
        id_list = group['id'].tolist()
        doc_list = group['token'].tolist()
        doc_list = list(map(lambda s: s.strip().replace('。', ' '), doc_list))
        res = {
            'id': id_list,
            'doc': doc_list,
        }
        with open(os.path.join(out_dir, str(cls) + '.pkl'), 'wb') as fp:
            pickle.dump(res, fp)


def read_data_from_dir(dir='data/trainSet/rank/tfidf/'):
    # file_lists = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    # for file in file_lists:
    #     cls, tp = file.split('_')
    #     if tp == 'id':
    #         print(cls)
    #         with open(os.path.join(dir, cls+'_id'), 'rb') as fp:
    #             docid_list = pickle.load(fp)
    #         with open(os.path.join(dir, cls+'_doc'), 'rb') as fp:
    #             corpus = pickle.load(fp)
    #         yield cls, docid_list, corpus

    #     names = ['9001', '9012', '9047', '9130', '9299',
    #              '9461', '9483', '9542', '9705', '9771',]
    names = ['9001', '9012', '9047', '9299',
             '9461', '9483', '9542', '9705', '9771',]

    for name in names:
        print(name)
        with open(os.path.join(dir, name + '.pkl'), 'rb') as fp:
            data = pickle.load(fp)
        yield name, data['id'], data['doc']


class TfIdfVec():
    def __init__(self, corpus, docid_list, cls, min_count=None, num_limit=5000, word_dic_path=None):
        self.corpus = corpus
        self.docid_list = docid_list
        self.cls = cls
        self.col = db['tfidf_' + cls]
        self.col.create_index([('word', ASCENDING)])
        self.min_count = min_count
        self.num_limit = num_limit
        self.word_dict_path = word_dic_path

    def __word_count(self):
        if self.word_dict_path:
            with open(self.word_dict_path, 'rb') as fp:
                word_dict = pickle.load(fp)
            word_dict = {word:index-1 for word,index in word_dict.items()}
            self.word_list = {index:word for word,index in word_dict.items()}
            return word_dict

        corpus = ' '.join(self.corpus)
        word_counts = Counter(filter(lambda word: len(word) > 1, corpus.split(' '))).items()
        word_counts_list = sorted(word_counts, key=lambda x: x[1], reverse=True)
        if self.min_count:
            word_counts_list = filter(lambda x: x[1] >= self.min_count, word_counts_list)
        else:
            word_counts_list = word_counts_list[:len(word_counts_list) * 0.99]
        self.word_list = list(map(lambda x: x[0], word_counts_list))
        word_dict = {word: index for index, word in enumerate(self.word_list)}
        return word_dict

    def fit(self):
        word_dict = self.__word_count()
        tfidfvector = TfidfVectorizer(vocabulary=word_dict)
        matrix = tfidfvector.fit_transform(self.corpus)
        print(len(tfidfvector.get_feature_names()))
        return matrix

    def save_db(self, matrix):
        nonzero = matrix.nonzero()
        dic = {
            'doc': nonzero[0],
            'word': nonzero[1],
            'tfidf': matrix.data,
        }
        df = pd.DataFrame(dic)
        buffer = []

        for word_index, group in df.groupby('word'):
            tmp = group.sort_values(by="tfidf", ascending=False)
            buffer.append({
                "word": self.word_list[word_index],
                "doc_tfidf": [{'doc': self.docid_list[doc_index], 'tfidf': tfidf} for doc_index, tfidf in
                              zip(tmp['doc'].tolist()[:self.num_limit], tmp['tfidf'].tolist()[:self.num_limit])]
            })

            if len(buffer) > 1000:
                self.col.insert(buffer)
                buffer.clear()
        if len(buffer) > 0:
            self.col.insert(buffer)
            buffer.clear()

    def run(self):
        matrix = self.fit()
        print(len(self.word_list))
        self.save_db(matrix)
        print(self.cls, ' finished!')


if __name__ == '__main__':
    trans_data_to_file(path='../data/data.csv', out_dir='../data/trainSet/rank/tfidf/')

    for cls, docid_list, corpus in read_data_from_dir(dir='../data/trainSet/rank/tfidf/'):
        tiv = TfIdfVec(corpus, docid_list, cls, min_count=30, word_dic_path=os.path.join('../data/trainSet/rank/dict/', cls+'_dictionary.dic'))
        tiv.run()

    print('Finished!')

