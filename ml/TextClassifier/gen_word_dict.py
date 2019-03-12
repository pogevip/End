import pickle
import pandas as pd
from collections import Counter
import os


def load_data(path):
    all_data = pd.read_csv(path)
    return all_data


class WordCounter:
    def __init__(self, data):
        self.limit = len(data)*0.001+1
        self.data = data

    def __word_count(self):
        corpus = ' '.join(self.data)
        corpus = corpus.replace('。', ' ')
        word_counts = Counter(corpus.split(' ')).items()
        word_wrap = filter(lambda x: x[1]>=self.limit, word_counts)
        word_set = set([x[0] for x in word_wrap])
        print('word_set size: ', len(word_set))
        return word_set

    def gen_word_set(self):
        return self.__word_count()

def word_dict_gengrator(src_data, to_path):
    print('load data')
    data = load_data(src_data)
    print('load finished!')

    word_dict = set()
    for _, group in data.groupby('cls'):
        print(_)
        wc = WordCounter(group['token'].tolist())

        word_dict |= wc.gen_word_set()

    word_dict = {word: index+1 for index, word in enumerate(list(word_dict))}

    print(len(word_dict))
    with open(to_path, 'wb') as fp:
        pickle.dump(word_dict, fp)
    print('finished!')


if __name__ == '__main__':
    # from_path = 'data/all_info.csv'
    # to_path = 'data/trainSet/classifier/dictionary.dic'
    # word_dict_gengrator(from_path, to_path)

    from_path = 'data/trainSet/rank/dict/'
    to_path = 'data/trainSet/classifier/dictionary.dic'

    word_dict = set()

    for f in os.listdir(from_path):
        f_path = os.path.join(from_path, f)
        if os.path.isdir(f_path):
            continue
        else:
            with open(f_path, 'rb') as fp:
                d = pickle.load(fp)
                d = set(d)
                word_dict |= d

    word_dict = {word: index + 1 for index, word in enumerate(list(word_dict))}
    print(len(word_dict))
    with open(to_path, 'wb') as fp:
        pickle.dump(word_dict, fp)
    print('finished!')

