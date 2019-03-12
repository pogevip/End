from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import Levenshtein
from pymongo import MongoClient
from collections import defaultdict
import re

conn = MongoClient('172.19.241.248', 20000)
col = conn['wangxiao']['alldata']

rep = re.compile('[A-Za-z0-9\&\《\》\〈\〉\﹤\﹥\、]')


def load_data():
    statutes_std = {
        '9001' : defaultdict(lambda: 0),
        '9012': defaultdict(lambda: 0),
        '9047': defaultdict(lambda: 0),
        '9130': defaultdict(lambda: 0),
        '9299': defaultdict(lambda: 0),
        '9461': defaultdict(lambda: 0),
        '9483': defaultdict(lambda: 0),
        '9542': defaultdict(lambda: 0),
        '9705': defaultdict(lambda: 0),
        '9771': defaultdict(lambda: 0),
    }

    demo = col.find(no_cursor_timeout=True)
    i = 0
    for item in demo:
        if i % 50000 == 0:
            print(i)
        try:
            for r in item['reference']:
                statutes_std[item['cls']][rep.sub('', r['name'])] += 1
        except:
            continue
        i += 1
    demo.close()

    for cls, status in statutes_std.items():
        tmp = [(k, v) for k, v in dict(status).items()]
        tmp.sort(key=lambda x: x[1], reverse=True)
        statutes_std[key] = tmp

    with open('data/statutes_count.pkl', 'wb') as fp:
        pickle.dump(statutes_std, fp)
    print('finished')


def read_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    for cls, status in data.items():
        corpus = [x[0] for x in status]
        yield cls, corpus


class StatutesStd:
    def __init__(self, corpus):
        self.corpus = [[s, {s}] for s in corpus]

    def __gen_statutes_vector(self):
        new_corpus = [' '.join(s[0]) for s in self.corpus]
        tfidfvector = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.matrix = tfidfvector.fit_transform(new_corpus)

    def __compute_cos_sim(self):
        self.__gen_statutes_vector()
        self.cosine_simlarity = cosine_similarity(self.matrix)

    def __edit_dis(self, str1, str2s, cos_sims):
        res = []
        for str2, cos_sim in zip(str2s, cos_sims):
            if cos_sim > 0:
                res.append(Levenshtein.ratio(str1, str2))
            else:
                res.append(0)
        return res

    def std(self, alpha=0.85):
        self.__compute_cos_sim()
        corpus = [s[0] for s in self.corpus]
        for i in range(len(corpus) - 1, 0, -1):
            cos_sims = list(self.cosine_simlarity[i][0:i])

            edit_sims = self.__edit_dis(corpus[i], corpus[0:i], cos_sims)

            sims = [(index, (cs + es) / 2) for index, cs, es in zip(range(i), cos_sims, edit_sims)]
            tmp = max(sims, key=lambda x:x[1])
            if tmp[1] > alpha:
                index = tmp[0]
                self.corpus[index][1] |= self.corpus[i][1]
                self.corpus.pop(i)

            if i % 100 == 0:
                print(i)
        print(len(self.corpus))

    def res(self):
        res = dict()
        for item in self.corpus:
            for key in item[1]:
                res[key] = item[0]



def process_all_data(col = 'alldata'):
    from_col = conn['wangxiao'][col]
    from_col.rename(col+'tmp')
    from_col = conn['wangxiao'][col+'tmp']

    out_col = conn['wangxiao']['alldata']

    with open('data/statutes_std.pkl', 'rb') as fp:
        statutes_std = pickle.load(fp)

    buffer = []

    demo = from_col.find(no_cursor_timeout=True)
    i = 0
    for item in demo:
        if i % 200000 == 0:
            print(i)
        try:
            tmp = item
            tmp_ref = []
            for r in item['reference']:
                name = statutes_std[item['cls']][rep.sub('', r['name'])]
                tmp_ref.append({'name' : name,
                            'levelone' : r['levelone'],
                            'leveltwo' : r['leveltwo']})
            tmp['reference'] = tmp_ref
            buffer.append(tmp)
            if len(buffer) >= 100000:
                out_col.insert_many(buffer)
                buffer.clear()
        except:
            continue
        i += 1
    demo.close()

    if len(buffer)>0:
        out_col.insert_many(buffer)
        buffer.clear()


if __name__ == '__main__':
    res = dict()
    for cls, corpus in read_data('data/statutes_count.pkl'):
        print(cls)
        ss = StatutesStd(corpus)
        ss.std(alpha=0.85)
        r = ss.res()
        res[cls] = r


    with open('data/statutes_std.pkl', 'wb') as fp:
        pickle.dump(res, fp)