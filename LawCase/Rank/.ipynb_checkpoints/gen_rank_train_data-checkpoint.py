# -*- coding: utf-8 -*-

from pymongo import MongoClient
import os
from collections import defaultdict
from math import log
from collections import Counter
import itertools
import pandas as pd
import pickle

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("rank_log.txt", mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

conn = MongoClient('172.19.241.248', 20000)

# CLS = ['9001', '9012', '9047', '9130', '9299',
#        '9461', '9483', '9542', '9705', '9771']
CLS = ['9047']


class RankTrainDataGen:
    def __init__(self, cls, data_path, logger):
        self.cls = cls
        self.tfidf_weight_col = conn['wangxiao']['web_tfidf_' + cls]
        self.data = pd.read_csv(data_path)
        self.logger = logger
        self.logger.info('data size : {}'.format(len(self.data)))

    def gen_train_data(self, max_count=25000, min_count=10000):
        res = list()

#         count = max(min(len(self.data) // 5, msax_count), min_count)
#         if len(self.data) > count:
#             data = self.data[(self.data['lenth']>=30) & (self.data['lenth']<=100)]
#             if len(data) > count:
#                 data = data.sample(count)
#         else:
#             data = self.data
            
        if len(self.data) > max_count:
            data = self.data.sample(max_count)
        else:
            data = self.data

        self.logger.info("data : {}".format(len(data)))
        
        def helper(raw):
            tmp = defaultdict(lambda: 0)
            for word in raw['token'].replace('。', ' ').split(' '):
                find_res = self.tfidf_weight_col.find_one({'word': word})
                if find_res:
                    for r in find_res['doc_tfidf']:
                        tmp[r['doc']] += r['tfidf']
            tmp = [(k, v) for k, v in dict(tmp).items()]
            tmp.sort(key=lambda x: x[1], reverse=True)
            tmp = tmp[:100]
            for t in tmp:
                res.append([raw['id'], t[0], t[1]])
            return
        
        data.apply(helper, axis=1)

#         i = 0
#         for _, item in data.iterrows():
#             # logger.info("item : {}".format(item['id']))
#             # 返回权重最大的100个案件
#             # logger.info("enter search")
#             tmp = defaultdict(lambda: 0)
#             try:
#                 for word in item['token'].replace('。', ' ').split(' '):
#                     find_res = self.tfidf_weight_col.find_one({'word': word})
#                     if find_res:
#                         for r in find_res['doc_tfidf']:
#                             tmp[r['doc']] += r['tfidf']
#             except:
#                 continue

#             tmp = [(k, v) for k, v in dict(tmp).items()]
#             tmp.sort(key=lambda x: x[1], reverse=True)
#             tmp = tmp[:100]
            
#             for t in tmp:
#                 res.append([item['id'], t[0], t[1]])

#             # 记录进度
#             i += 1
#             if i % 100 == 0:
#                 # print(i)
#                 self.logger.info("--step : {}".format(i))

        return pd.DataFrame(res, columns=['query', 'doc', 'tfidf'])

    def run(self):
        # print('----gen train data')
        self.logger.info('----gen train data')
        self.res = self.gen_train_data()

    def out(self, out_dir):
        # print('----write data')
        self.logger.info('----write data')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, self.cls + '.csv')
        self.res.to_csv(out_path, index=0)

if __name__ == '__main__':
    logger.info('read_data')
    for cls in CLS:
        logger.info("-----cls--: {}".format(cls))
        rtd = RankTrainDataGen(cls, os.path.join('../data/trainSet/rank/each_cls_data_web', cls+'.csv'), logger)
        rtd.run()
        rtd.out('../data/trainSet/rank/search_res_web')
    logger.info('finished')