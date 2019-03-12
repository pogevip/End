# coding:utf-8

import os
import pickle
import pandas as pd
from pymongo import MongoClient
from gensim.models import LdaModel
from gensim.corpora import Dictionary

conn = MongoClient('172.19.241.248', 20000)
db = conn['wangxiao']


def split_data(path='data/data.csv', out_path='data/trainSet/search/testdata/'):
    df = pd.read_csv(path)
    data = df[df['cls'] == 9047].sample(n=100100)
    data['tag'] = 1

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    test_index = data.sample(n=100).index
    data.loc[test_index, 'tag'] = 0
    test = {
        'id': data.loc[test_index, 'id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data.loc[test_index, 'token'].tolist()))
    }
    with open(os.path.join(out_path, 'test.pkl'), 'wb') as fp:
        print(len(test['id']))
        pickle.dump(test, fp)

    group10w = {
        'id': data[data['tag'] == 1]['id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data[data['tag'] == 1]['token'].tolist()))
    }
    with open(os.path.join(out_path, 'group10w.pkl'), 'wb') as fp:
        print(len(group10w['id']))
        pickle.dump(group10w, fp)

    group5w_index = data[data['tag'] == 1].sample(n=50000).index
    data.loc[group5w_index, 'tag'] = 2
    group5w = {
        'id': data[data['tag'] == 2]['id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data[data['tag'] == 2]['token'].tolist()))
    }
    with open(os.path.join(out_path, 'group5w.pkl'), 'wb') as fp:
        print(len(group5w['id']))
        pickle.dump(group5w, fp)

    group1w_index = data[data['tag'] == 2].sample(n=10000).index
    data.loc[group1w_index, 'tag'] = 3
    group1w = {
        'id': data[data['tag'] == 3]['id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data[data['tag'] == 3]['token'].tolist()))
    }
    with open(os.path.join(out_path, 'group1w.pkl'), 'wb') as fp:
        print(len(group1w['id']))
        pickle.dump(group1w, fp)

    group5k_index = data[data['tag'] == 3].sample(n=5000).index
    data.loc[group5k_index, 'tag'] = 4
    group5k = {
        'id': data[data['tag'] == 4]['id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data[data['tag'] == 4]['token'].tolist()))
    }
    with open(os.path.join(out_path, 'group5k.pkl'), 'wb') as fp:
        print(len(group5k['id']))
        pickle.dump(group5k, fp)

    group1k_index = data[data['tag'] == 4].sample(n=1000).index
    data.loc[group1k_index, 'tag'] = 5
    group1k = {
        'id': data[data['tag'] == 5]['id'].tolist(),
        'doc': list(map(lambda s: s.strip().replace('。', ' '),
                        data[data['tag'] == 5]['token'].tolist()))
    }
    with open(os.path.join(out_path, 'group1k.pkl'), 'wb') as fp:
        print(len(group1k['id']))
        pickle.dump(group1k, fp)

    print('finish')


class LDAVecGen():
    def __init__(self, path):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        self.id_list = data['id']
        self.doc_list = list(map(lambda doc: doc.split(' '), data['doc']))

    def fit_model(self, topic_num):
        self.dictionary = Dictionary(self.doc_list)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.doc_list]
        self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=topic_num)

    def out(self, model_path, col_name):
        buffer = []
        vecs = self.model.get_document_topics(self.corpus)

        col = db[col_name]
        for id, vec in zip(self.id_list, vecs):
            buffer.append({
                'fulltextid': id,
                'vec': [[item[0], float(item[1])] for item in vec]
            })

            if len(buffer) >= 1000:
                col.insert_many(buffer)
                buffer.clear()

        if len(buffer) > 0:
            col.insert_many(buffer)
            buffer.clear()

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.dictionary.save(os.path.join(model_path, 'lda.dic'))
        # loaded_dct = Dictionary.load(tmp_fname)
        self.model.save(os.path.join(model_path, 'lda.model'))
        # model = LdaModel.load(fname, mmap='r')




if __name__ == '__main__':
    split_data(path='../data/data.csv', out_path='../data/trainSet/search/testdata/')