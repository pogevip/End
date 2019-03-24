import pandas as pd
import pickle
import os


def load_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    return dic


def load_cls_dict(path):
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
    dic = {cls: item[0] for cls, item in dic.items()}
    return dic


def load_data(path):
    data = pd.read_csv(path)
    data = data[~data['token'].isna()]
    return data


def gen_train_file(all_data, word_dic, cls_dic, out_path, is_cls=[]):
    #is_cls为1，则为分类存储，若为0则不分类
    all_data['X'] = all_data['token'].apply(lambda doc:
                                            list(map(lambda w: word_dic[w] if w in word_dic else 0, doc.split(' '))))
    all_data['Y'] = all_data['cls'].apply(lambda cls: cls_dic[cls])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for tag, group in all_data.groupby('train_val_test'):
        print(tag)
        if tag == 1:
            out = 'train.pkl'
        elif tag == 2:
            out = 'val.pkl'
        elif tag == 3:
            out = 'test.pkl'
        else:
            raise ValueError('tag error!')

        if tag in is_cls:
            #对训练集分类存储
            res = dict()
            for cls, tmp in group.groupby('cls'):
                X = tmp['X'].tolist()
                Y = tmp['Y'].tolist()
                res[cls] = {
                    'X': X,
                    'Y': Y
                }
        else:
            tmp = group.sample(frac=1).reset_index(drop=True)

            X = tmp['X'].tolist()
            Y = tmp['Y'].tolist()

            res = {
                'X': X,
                'Y': Y
            }

        with open(os.path.join(out_path, out), 'wb') as fp:
            pickle.dump(res, fp)

    print('finish!')


if __name__ == '__main__':
    out_path = '../data/trainSet/classifier/train_file_5w'
    data_path = '../data/trainSet/classifier/train_info_5w.csv'
    word_dict_path = '../data/trainSet/classifier/dictionary.dic'
    cls_dict_path = '../data/trainSet/classifier/cls_5w.dic'

    all_data = load_data(data_path)
    word_dict = load_dict(word_dict_path)
    cls_dict = load_cls_dict(cls_dict_path)

    gen_train_file(all_data, word_dict, cls_dict, out_path, is_cls=[])