import pandas as pd
import os
import pickle

from itertools import chain

# CLS = ['9001', '9012', '9047', '9130', '9299',
#        '9461', '9483', '9542', '9705', '9771']

b_statute = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
b_case = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def reset_dic(option):
    dic = {}

    b = b_case if option=='c' else b_statute

    for i in b:
        dic[i] = {
            'p': [],
            'r': [],
            'f': [],
        }
    return dic


def test(cls, rank_path, ref_path, out_path):
    rank_data = pd.read_csv(os.path.join(rank_path, cls + '.csv'))
    ref_data = pd.read_csv(os.path.join(ref_path, cls + '.csv'))

    data = rank_data.merge(ref_data, left_on='query', right_on='id').merge(ref_data, left_on='doc', right_on='id')

    data = data.loc[:, ['query', 'ref_x', 'ref_y', 'sim', 'statute_sim']]
    
    res = reset_dic('c')
#     res = reset_dic('s')

    global flag
    flag = False

    def statute_helper(df):
        global flag
        if flag == False:
            flag = True
            return

        statutes1_tmp = df['ref_x'].tolist()
        if len(statutes1_tmp) == 0:
            return
        statutes1 = set(chain(*map(lambda x: x.split('--'), statutes1_tmp)))

        for b in b_statute:
            tmp = df[df['statute_sim'] >= b]
            statutes2_tmp = tmp['ref_y'].tolist()

            b_tmp = b
            attenuation = 0
            while len(statutes2_tmp) == 0:
                if b_tmp < 0:
                    res[b]['p'].append(0.)
                    res[b]['r'].append(0.)
                    res[b]['f'].append(0.)
                    return
                b_tmp -= 0.1
                attenuation += 0.1
                tmp = df[df['statute_sim'] >= b_tmp]
                statutes2_tmp = tmp['ref_y'].tolist()

            statutes2 = set(chain(*map(lambda x: x.split('--'), statutes2_tmp)))

            in_set = statutes1 & statutes2

            p = len(in_set) / len(statutes2)
            r = len(in_set) / len(statutes1) - attenuation
            r = max(0,r)
            f = 2 * p * r / (p + r) if (p + r) != 0 else 0

            res[b]['p'].append(p)
            res[b]['r'].append(r)
            res[b]['f'].append(f)
        return

    def case_helper(df):
        global flag
        if flag == False:
            flag = True
            return
        
        statutes1_tmp = df['ref_x'].tolist()
        if len(statutes1_tmp) == 0:
            return
        statutes1 = set(chain(*map(lambda x: x.split('--'), statutes1_tmp)))

        statutes2_all = df.sort_values(by="sim", ascending=False)['ref_y'].tolist()

        for b in b_case:
            statutes2_tmp = statutes2_all[:b]
            statutes2 = set(chain(*map(lambda x: x.split('--'), statutes2_tmp)))

            in_set = statutes1 & statutes2

            p = len(in_set) / len(statutes2)
            r = len(in_set) / len(statutes1)
            f = 2 * p * r / (p + r) if (p + r) != 0 else 0

            res[b]['p'].append(p)
            res[b]['r'].append(r)
            res[b]['f'].append(f)
        return

    data.groupby('query').apply(case_helper)
#     data.groupby('query').apply(statute_helper)


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, cls + '.pkl'), 'wb') as fp:
        pickle.dump(res, fp)

    print('finish!')


if __name__ == '__main__':
    rank_data_dir = '../data/trainSet/rank/search_res_std/'
    ref_data_dir = '../data/trainSet/rank/each_cls_data/'
    out_dir = '../data/trainSet/rank/test_case/'

    CLS = ['9771', '9001', '9012', '9047', '9299',
           '9461', '9483', '9542', '9705', ]

    for cls in CLS:
        print('------' + cls + '------')
        test(cls, rank_data_dir, ref_data_dir, out_dir)
