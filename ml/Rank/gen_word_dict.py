import pickle
import pandas as pd
from collections import Counter
import os

def load_data(path):
    all_data = pd.read_csv(path)
    return all_data


class WordCounter:
    def __init__(self, data):
        self.limit = min(len(data)*0.0001+1, 50)
        self.data = data

    def __word_count(self):
        corpus = ' '.join(self.data)
        corpus = corpus.replace('。', ' ')
        word_counts = Counter(corpus.split(' ')).items()
        print(len(word_counts))
        word_wrap = filter(lambda x: x[1]>=self.limit, word_counts)
        word_set = set([x[0] for x in word_wrap])
        print('word_set size: ', len(word_set))
        return word_set

    def gen_word_dict(self, path):
        word_dict = self.__word_count()
        word_dict = {word: index + 1 for index, word in enumerate(list(word_dict))}

        with open(path, 'wb') as fp:
            pickle.dump(word_dict, fp)
        print('finished')


def word_dict_gengrator(src_data, to_path):
    print('load data')
    data = load_data(src_data)
    print('load finished!')

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    for cls, group in data.groupby('cls'):
        print(cls)
        if str(cls) == '9103':
            continue
        wc = WordCounter(group['token'].tolist())

        wc.gen_word_dict(os.path.join(to_path, str(cls)+'_dictionary.dic'))

    print('finished!')


if __name__ == '__main__':
    from_path = 'data/data.csv'
    to_path = 'data/trainSet/rank/dict/'
    word_dict_gengrator(from_path, to_path)