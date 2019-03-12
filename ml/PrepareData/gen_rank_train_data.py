from pymongo import MongoClient
import os
from collections import defaultdict
from math import log
from collections import Counter
import itertools
import pandas as pd

# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler("rank_log.txt", mode='w')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

conn = MongoClient('172.19.241.248', 20000)

CLS = ['9001', '9012', '9047', '9130', '9299',
       '9461', '9483', '9542', '9705', '9771']



def load_data():
    all_data = pd.read_csv('../data/trainSet/rank/data.csv')
    return all_data


class RankTrainDataGen:
    def __init__(self, data, cls, logger):
        self.cls = cls
        self.tfidf_weight_col = conn['wangxiao']['tfidf_' + cls]
        self.data = data
        self.logger = logger
        self.logger.info('data size : {}'.format(len(self.data)))
        self.statutes_weight = dict()

    def gen_statutes_weight(self):
        statutes = self.data['ref'].tolist()
        statutes = filter(lambda x: x != '' and x, statutes)
        statutes = map(lambda x: x.split('--'), statutes)
        statutes = itertools.chain(*statutes)
        statutes_count = Counter(statutes)

        all_count = len(self.data)
        for k, v in statutes_count.items():
            self.statutes_weight[k] = 1 + log((all_count + 1) / (v + 1))

    def __gen_sim(self, statutes1, statutes2):
        in_set = statutes1 & statutes2
        un_set = statutes1 | statutes2
        a = sum([self.statutes_weight[s] for s in in_set])
        b = sum([self.statutes_weight[s] for s in un_set])
        return a / b

    def gen_train_data(self, max_count=100000, min_count=40000):
        res = list()

        count = max(min(len(self.data) // 5, max_count), min_count)
        if len(self.data) > count:
            data = self.data.sample(count)
        else:
            data = self.data

        self.logger.info("data : {}".format(len(data)))

        i = 0
        for _, item in data.iterrows():
            # logger.info("item : {}".format(item['id']))

            # 返回权重最大的100个案件
            # logger.info("enter search")
            tmp = defaultdict(lambda: 0)
            try:
                for word in item['token'].replace('。', ' ').split(' '):
                    find_res = self.tfidf_weight_col.find_one({'word': word})
                    if find_res:
                        for r in find_res['doc_tfidf']:
                            tmp[r['doc']] += r['tfidf']
            except:
                continue

            tmp = [(k, v) for k, v in dict(tmp).items()]
            tmp.sort(key=lambda x: x[1], reverse=True)
            tmp = tmp[:100]

            # 记录两个案件的id并计算法条相似度
            # logger.info("enter sim")
            for t in tmp:
                try:
                    find_res = self.data.loc[self.data['id'] == t[0]]['ref'].iloc[0]
                    refs = set(find_res.split('--'))
                    sim = self.__gen_sim(set(item['ref'].split('--')), refs)
                    self.logger.info("{}-{}-{}".format(item['id'], t[0], sim))
                    res.append([item['id'], t[0], sim])
                except:
                    continue

            #记录进度
            i+=1
            if i % 30 == 0:
                # print(i)
                self.logger.info("--step : {}".format(i))

        return pd.DataFrame(self.res, columns=['query', 'doc', 'sim'])

    def run(self):
        # print('---- compute statutes weight')
        self.logger.info('---- compute statutes weight')
        self.gen_statutes_weight()
        # print('----gen train data')
        self.logger.info('----gen train data')
        self.res = self.gen_train_data()

    def out(self, out_dir='data/trainSet/rank'):
        # print('----write data')
        self.logger.info('----write data')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, self.cls + '_all.csv')
        self.res.to_csv(out_path, index=0)


# if __name__ == '__main__':
#     #     print('read data..')
#     logger.info('read_data')
#     all_data = load_data()
#     for cls in CLS:
#         #         print(cls)
#         logger.info(cls)
#         rtd = RankTrainDataGen(all_data, cls, logger)
#         rtd.run()
#         rtd.out('../data/trainSet/rank')
#     logger.info('finished')