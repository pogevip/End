import pandas as pd
import pickle
import os
import itertools
from collections import Counter


# CLS = ['9001', '9012', '9047', '9130', '9299',
#        '9461', '9483', '9542', '9705', '9771']

def gen_statutes_weight(data, cls):
    statutes = data['ref'].tolist()
    statutes = filter(lambda x: x != '' and x, statutes)
    statutes = map(lambda x: x.split('--'), statutes)
    statutes = itertools.chain(*statutes)
    statutes_count = Counter(statutes)

    statutes_weight = dict()
    all_count = len(data)
    print(all_count)
    for k, v in statutes_count.items():
        statutes_weight[k] = 1 + log((all_count + 1) / (v + 1))

    with open(os.path.join('data/trainSet/rank/statute_weight/', cls + '_weight.pkl'), 'wb') as fp:
        pickle.dump(statutes_weight, fp)


def std_tfidf(df):
    max = df['tfidf'].max()
    min = df['tfidf'].min()
    df['tfidf_sim'] = (df['tfidf']-min)/(max-min)
    return df


def gen_statute_sim(statutes1, statutes2, statutes_weight):
    in_set = statutes1 & statutes2
    un_set = statutes1 | statutes2
    a = sum([statutes_weight[s] for s in in_set])
    b = sum([statutes_weight[s] for s in un_set])
    return a / b


def gen_sim(df, statutes_weight, alpha=0.3):
    statutes1 = set(df['ref_x'].split('--'))
    statutes2 = set(df['ref_y'].split('--'))
    df['statute_sim'] = gen_statute_sim(statutes1, statutes2, statutes_weight)
    df['sim'] = alpha*df['tfidf_sim']+(1-alpha)*df['statute_sim']
    return df


def com_sim(rank_data_path, info_data_path, statutes_weight_path, out_path):
    rank_data = pd.read_csv(rank_data_path)
    info_data = pd.read_csv(info_data_path)
    with open(statutes_weight_path, 'rb') as fp:
        statutes_weight = pickle.load(fp)

    print(len(rank_data))

    print('merge...')
    data = rank_data.merge(info_data, left_on='query', right_on='id').merge(info_data, left_on='doc', right_on='id')
    data = data.loc[:, ['query', 'doc', 'ref_x', 'ref_y', 'tfidf']]
#     print(data.head(5))

    print('std tfidf sim...')
    data = data.groupby('query').apply(std_tfidf)
#     print(data.head(5))

    
    print('compute statute sim and sim')
    data = data.apply(gen_sim, statutes_weight=statutes_weight, axis=1)

    data = data.loc[:, ['query', 'doc', 'tfidf_sim', 'statute_sim', 'sim']]

    print('write...')
    data.to_csv(out_path, index=0)

    print('finish!')


if __name__ == '__main__':
    rank_data_dir = '../data/trainSet/rank/search_res_web/'
    info_data_dir = '../data/trainSet/rank/each_cls_data_web/'
    statutes_weight_dir = '../data/trainSet/rank/statute_weight/'
    out_data_dir = '../data/trainSet/rank/search_res_std_web/'

    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

#     CLS = ['9001', '9012', '9047', '9130', '9299', '9461', '9483', '9542', '9705', '9771']
    CLS = ['9047', '9130', '9542', '9705']

    for cls in CLS:
        print('------' + cls + '------')
        com_sim(rank_data_path=os.path.join(rank_data_dir, cls+'.csv'),
                info_data_path=os.path.join(info_data_dir, cls+'.csv'),
                statutes_weight_path=os.path.join(statutes_weight_dir, cls+'_weight.pkl'),
                out_path=os.path.join(out_data_dir, cls+'.csv'),)