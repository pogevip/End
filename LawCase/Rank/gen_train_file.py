import pandas as pd
import pickle
import os
import math


# CLS = ['9001', '9012', '9047', '9130', '9299',
#        '9461', '9483', '9542', '9705', '9771']

def load_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    return dic


def load_data(path):
    df = pd.read_csv(path)
    return df


def del_helper(df):
    tmp = df[df['statute_sim'] == 0]
    del_num = len(tmp) - math.ceil((len(df) - len(tmp)) * 0.2)
    if del_num > 0:
        del_index = tmp.sample(del_num).index
        df.drop(index=del_index, inplace=True)
    return df



def gen_train_data(rank_data_path, info_data_path, word_dic):
    print('load data...')
    rank_data = load_data(rank_data_path)
    info_data = load_data(info_data_path)

    print(len(rank_data))

    print('delete zero..')
    rank_data = rank_data.groupby('query').apply(del_helper)
    rank_data.reset_index(drop=True, inplace=True)
    print(len(rank_data))

    print('merge...')
    data = rank_data.merge(info_data, left_on='query', right_on='id').merge(info_data, left_on='doc', right_on='id')
    data = data.loc[:, ['token_x', 'token_y', 'statute_sim']]
    print(len(data))

    print('code...')
    def code_helper(doc):
        doc_list = list(map(lambda w: str(word_dic[w]) if w in word_dic else 0, doc.replace('。', ' ').split(' ')))
        if len(doc_list)>400:
            doc_list = list(filter(lambda w: w!=0, doc_list))
#         return '--'.join(doc_list)
        return ' '.join([str(x) for x in doc_list])
    
    data['X1'] = data['token_x'].apply(code_helper)
    data['X2'] = data['token_y'].apply(code_helper)

    print('split...')
    data['train_val_test'] = 1

    test_index = data.sample(frac=0.15).index
    data.loc[test_index, 'train_val_test'] = 3

    val_index = data[data['train_val_test'] == 1].sample(frac=0.15).index
    data.loc[val_index, 'train_val_test'] = 2

    data = data.loc[:, ['X1', 'X2', 'statute_sim', 'train_val_test']]
    
#     print('write...')
#     data.to_csv(out_dir+'tmp/'+cls+'.csv', index=0)

    return data


def write_to_file(data, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
#     print('process...')
#     data['X1'] = data['X1'].apply(lambda x: ','.join(x.split('--')[:200]))
#     data['X2'] = data['X2'].apply(lambda x: ','.join(x.split('--')[:200]))
    
    print('write...')
    for tag, group in data.groupby('train_val_test'):
        print(tag)
        if tag == 1:
            out = 'train.pkl'
        elif tag == 2:
            out = 'val.pkl'
        elif tag == 3:
            out = 'test.pkl'
        else:
            raise ValueError('tag error!')

        tmp = group.sample(frac=1).reset_index(drop=True)

        query = tmp['X1'].tolist()
        doc = tmp['X2'].tolist()
        sim = tmp['statute_sim'].tolist()

        res = {
            'X1': query,
            'X2': doc,
            'Y': sim,
        }

        with open(os.path.join(out_path, out), 'wb') as fp:
            pickle.dump(res, fp)

    print('finish!')


if __name__ == '__main__':
    word_dic_dir = '../data/trainSet/rank/dict/'
    info_data_dir = '../data/trainSet/rank/web_each_cls_data/'
    rank_data_dir = '../data/trainSet/rank/web_search_res_std/'

    out_dir = '../data/trainSet/rank/web_train_file/'

    CLS = ['9001', '9012', '9047', '9130', '9299',
           '9461', '9483', '9542', '9705', '9771']

    for cls in CLS:
        print('------' + cls + '------')
        word_dic = load_dict(os.path.join(word_dic_dir, cls + '_dictionary.dic'))
        data = gen_train_data(rank_data_path=os.path.join(rank_data_dir, cls + '.csv'),
                              info_data_path=os.path.join(info_data_dir, cls + '.csv'),
                              word_dic=word_dic)
        write_to_file(data, out_dir + cls + '/')
