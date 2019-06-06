from pymongo import MongoClient
import pickle
import os


class WordVec:
    def __init__(self):
        self.words = set()

    def get_all_words(self, col):
        demo = col.find(no_cursor_timeout=True)
        index = 0
        for item in demo:
            if index % 50000 == 0:
                print(index)
            index += 1
            self.words |= set(item['text'])
        demo.close()

    def gen_vec(self, out_path):
        dir_name = os.path.dirname(out_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        dim = 1
        x = 1
        while x * 2 <= len(self.words):
            x *= 2
            dim += 1

        word_vec = dict()

        for index, word in enumerate(self.words):
            vec = list(map(int, reversed(str(bin(index))[2:])))
            vec.extend([0] * (dim - len(vec)))
            word_vec[word] = vec

        with open(out_path, 'wb') as fp:
            pickle.dump(word_vec, fp)

if __name__ == '__main__':
    # conn = MongoClient('172.19.241.248', 20000)
    # col = conn.wangxiao.alldata
    # wv = WordVec()
    # wv.get_all_words(col)
    # wv.gen_vec('')
    pass

