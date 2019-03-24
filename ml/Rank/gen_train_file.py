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


def gen_train_file(rank_data, info_data, word_dic, out_path):
    print(len(rank_data))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print('delete zero..')
    i = 0
    for _, group in rank_data.groupby('query'):
        tmp = group[group['statute_sim']==0]
        del_num = len(tmp) - math.ceil((len(group)-len(tmp))*0.2)
        if del_num > 0:
            del_index = tmp.sample(del_num).index
            rank_data.drop(index=del_index, inplace=True)
        if i%200 == 0:
            print(i)
        i+=1
    print(len(rank_data))
    
    print('merge...')
    data = rank_data.merge(info_data, left_on='query', right_on='id').merge(info_data, left_on='doc', right_on='id')
    data = data.loc[:, ['token_x', 'token_y', 'statute_sim']]
    
    print('code...')
    data['X1'] = data['token_x'].apply(lambda doc:
                                            list(map(lambda w: word_dic[w] if w in word_dic else 0, doc.replace('。', ' ').split(' '))))
    data['X2'] = data['token_y'].apply(lambda doc:
                                            list(map(lambda w: word_dic[w] if w in word_dic else 0, doc.replace('。', ' ').split(' '))))
    
    print('split...')
    data['train_val_test'] = 1

    test_index = data.sample(frac=0.15).index
    data.loc[test_index, 'train_val_test'] = 3

    val_index = data[data['train_val_test']==1].sample(frac=0.15).index
    data.loc[val_index, 'train_val_test'] = 2
    
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
            'Y' : sim,
        }

        with open(os.path.join(out_path, out), 'wb') as fp:
            pickle.dump(res, fp)

    print('finish!')


if __name__ == '__main__':
    word_dic_dir = '../data/trainSet/rank/dict/'
    info_data_dir = '../data/trainSet/rank/each_cls_data/'
    rank_data_dir = '../data/trainSet/rank/search_res/'
    out_dir = '../data/trainSet/rank/train_file/'
    
    CLS = ['9047', '9299', '9461', '9542', '9012', '9705']

    for cls in CLS:
        print('------'+cls+'------')
        word_dic = load_dict(os.path.join(word_dic_dir, cls+'_dictionary.dic'))
        info_data = load_data(os.path.join(info_data_dir, cls+'.csv'))
        rank_data = load_data(os.path.join(rank_data_dir, cls + '.csv'))
        gen_train_file(rank_data, info_data, word_dic, out_dir+cls+'/')