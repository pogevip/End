import pickle
import os
from pymongo import MongoClient
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from collections import defaultdict
import time

conn = MongoClient('172.19.241.248', 20000)
db = conn['wangxiao']

def load_test_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


class Comp:
    def __init__(self, test_data_path, option):
        with open(test_data_path, 'rb') as fp:
            data = pickle.load(fp)
        self.id_list = data['id']
        self.doc_list = list(map(lambda doc: doc.split(' '), data['doc']))

        self.all_col = db['alldata_final']
        self.option = option
        
        self.tfidf = {
            'time':list(),
            'precision': list(),
            'recall':list()
        }

        self.lda = {
            'time':list(),
            'precision': list(),
            'recall':list()
        }


    def __compute_vec_sim(self, vec1, vec2):
        if vec2[0][0] < vec1[0][0]:
            vec1, vec2 = vec2, vec1

        i, j = 0, 0
        mole, vec12, vec22 = 0, 0, 0
        while i < len(vec1) and j < len(vec2):
            if vec1[i][0] == vec2[j][0]:
                mole += vec1[i][1] * vec2[j][1]
                vec12 += vec1[i][1] ** 2
                vec22 += vec2[j][1] ** 2
                i += 1
                j += 1
            elif vec1[i][0] < vec2[j][0]:
                vec12 += vec1[i][1] ** 2
                i += 1
            else:
                vec22 += vec2[j][1] ** 2
                j += 1

        while i < len(vec1):
            vec12 += vec1[i][1] ** 2
            i += 1
        while j < len(vec2):
            vec22 += vec2[j][1] ** 2
            j += 1
        return mole / (vec12 * vec22)


    def __compute_precision_recall(self, id, search_id_list):
        find = self.all_col.find_one({'fullTextId': id})
        statute_std = set()
        if find:
            for s in find['reference']:
                statute_std.add(s['name']+s['levelone'])

        precision_list = []
        statute_find = set()
        for sid in search_id_list:
            find = self.all_col.find_one({'fullTextId': sid})
            statute_find_tmp = set()
            if find:
                for s in find['reference']:
                    tmp = s['name'] + s['levelone']
                    statute_find.add(tmp)
                    statute_find_tmp.add(tmp)
                if len(statute_find_tmp & statute_std)>0:
                    precision_list.append(1)
                else:
                    precision_list.append(0)

        precision = sum(precision_list)/len(precision_list)
        recall = len(statute_find & statute_std)/len(statute_std)
        return precision, recall


    def search_by_tfidf(self):
        col = db['tfidf_test_'+self.option]
        j = 0
        for id, doc in zip(self.id_list, self.doc_list):
            j+=1
            print(j)
            time_start = time.time()
            tmp = defaultdict(lambda: 0)
            for word in doc:
                find_res = col.find_one({'word': word})
                if find_res:
                    for r in find_res['doc_tfidf']:
                        tmp[r['doc']] += r['tfidf']

            tmp = [[id,sim] for id, sim in tmp.items()]
            tmp.sort(key=lambda x: x[1], reverse=True)
            tmp = tmp[:100]

            time_end = time.time()
            self.tfidf['time'].append([len(doc), time_end-time_start])

            precision, recall = self.__compute_precision_recall(id, [item[0] for item in tmp])
            self.tfidf['precision'].append(precision)
            self.tfidf['recall'].append(recall)


    def search_by_lda(self):
        dic = Dictionary.load(os.path.join('data/trainSet/search/lda_model/', self.option, 'lda.dic'))
        model = LdaModel.load(os.path.join('data/trainSet/search/lda_model/', self.option, 'lda.model'), mmap='r')
        col = db['ldavec_' + self.option]
        j=0
        for id, doc in zip(self.id_list, self.doc_list):
            j += 1
            print(j)

            time_start = time.time()
            tmp = []
            corpus = dic.doc2bow(doc)
            vec_std = model.get_document_topics(corpus)

            demo = col.find(no_cursor_timeout = True)
            for item in demo:
                vec = item['vec']
                sim = self.__compute_vec_sim(vec_std, vec)
                tmp.append([item['fulltextid'], sim])
            demo.close()

            tmp.sort(key=lambda x: x[1], reverse=True)
            tmp = tmp[:100]

            time_end = time.time()
            self.lda['time'].append([len(doc), time_end-time_start])

            precision, recall = self.__compute_precision_recall(id, [item[0] for item in tmp])
            self.lda['precision'].append(precision)
            self.lda['recall'].append(recall)

    def get_res(self, out_path, how = 'all'):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if how == 'tfidf':
            with open(os.path.join(out_path, 'tfidf_'+self.option+'_res.pkl'), 'wb') as fp:
                pickle.dump(self.tfidf, fp)
        elif how == 'lda':
            with open(os.path.join(out_path, 'lda_'+self.option+'_res.pkl'), 'wb') as fp:
                pickle.dump(self.lda, fp)
        else:
            with open(os.path.join(out_path, 'tfidf_'+self.option+'_res.pkl'), 'wb') as fp:
                pickle.dump(self.tfidf, fp)
            with open(os.path.join(out_path, 'lda_'+self.option+'_res.pkl'), 'wb') as fp:
                pickle.dump(self.lda, fp)