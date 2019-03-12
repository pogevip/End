import pickle
import pandas as pd


def load_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data

def load_all_data(path):
    all_data = pd.read_csv(path)
    return all_data


def split_train_val_test_set(self, frac=0.2, max_num=30000):
    # 划分测试集
    for _, group in self.res.groupby('cls'):
        if len(group) * frac <= max_num:
            test_index = group.sample(frac=frac).index
            self.res.loc[test_index, 'train_val_test'] = 3
        else:
            test_index = group.sample(max_num).index
            self.res.loc[test_index, 'train_val_test'] = 3

    # 划分验证集
    train_df = self.res[self.res['train_val_test'] == 1]
    for _, group in train_df.groupby('cls'):
        if len(group) * frac <= max_num:
            val_index = group.sample(frac=frac).index
            self.res.loc[val_index, 'train_val_test'] = 2
        else:
            val_index = group.sample(max_num).index
            self.res.loc[val_index, 'train_val_test'] = 2