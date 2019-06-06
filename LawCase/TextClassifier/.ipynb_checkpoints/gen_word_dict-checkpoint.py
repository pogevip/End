import pickle
import pandas as pd
from collections import Counter, defaultdict
import os


def load_data(path):
    all_data = pd.read_csv(path)
    return all_data


class WordCounter:
    def __init__(self, data, limit=None):
        if not limit:
            self.limit = min(len(data) * 0.0001 + 1, 50)
        else:
            self.limit = limit
        self.data = data

    def __word_count(self):
        word_dict = defaultdict(lambda : 0)
        corpus = map(lambda x:x.replace('。', ' ').split(' '), self.data)

        for doc in corpus:
            for w, c in Counter(doc).items():
                word_dict[w] += c

        word_dict = [[k, v] for k,v in dict(word_dict).items()]
        word_wrap = filter(lambda x: x[1] >= self.limit, word_dict)
        word_set = [x[0] for x in word_wrap]
        print('word_set size: ', len(word_set))
        return word_set

    def gen_word_dict(self, path):
        word_dict = self.__word_count()
        word_dict = {word: index + 1 for index, word in enumerate(word_dict)}

        with open(path, 'wb') as fp:
            pickle.dump(word_dict, fp)
        print('finished')


def word_dict_gengrator(src_data, to_path, option=None):
    print('load data')
    data = load_data(src_data)
    print('load finished!')

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    if option == 'cls':
        for cls, group in data.groupby('cls'):
            print(cls)
            wc = WordCounter(group['token'].tolist())
            wc.gen_word_dict(os.path.join(to_path, str(cls) + '_dictionary.dic'))
    else:
        wc = WordCounter(data['token'].tolist(), limit=20)
        wc.gen_word_dict(os.path.join(to_path, 'all_dictionary.dic'))

    print('finished!')


if __name__ == '__main__':
    from_path = 'data/data.csv'
    to_path = 'data/trainSet/rank/dict/'
    word_dict_gengrator(from_path, to_path)