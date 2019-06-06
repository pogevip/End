import pandas as pd
import os


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
        
def gen_sim(statutes1, statutes2):
    in_set = statutes1 & statutes2
    un_set = statutes1 | statutes2
    a = sum([self.statutes_weight[s] for s in in_set])
    b = sum([self.statutes_weight[s] for s in un_set])
    return a / b


def helper(df):
    max = df['tfidf_sim'].max()
    min = df['tfidf_sim'].min()
    df['tfidf_sim_std'] = (df['tfidf_sim']-min)/(max-min)
    df['sim'] = 0.3*df['tfidf_sim_std'] + 0.7*df['statute_sim']
    return df


def gen_train_file(cls, src_path, out_path):
    data = pd.read_csv(os.path.join(src_path, cls+'.csv'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data = data.groupby('query').apply(helper)

    data.to_csv(os.path.join(out_path, cls+'.csv'), index=0)

    print('finish!')


if __name__ == '__main__':
    rank_data_dir = '../data/trainSet/rank/search_res/'
    out_dir = '../data/trainSet/rank/search_res_std/'

    CLS = ['9012', '9705']

    for cls in CLS:
        print('------' + cls + '------')
        gen_train_file(cls, rank_data_dir, out_dir)