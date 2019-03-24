import jieba.posseg
from time import clock


class Token:
    def __init__(self, input):
        self.input = input
        self.stop_flags = ['b', 'c', 'e', 'g', 'h', 'k', 'l', 'o', 's',
                           'u', 'w', 'x', 'y', 'z', 'un', 'nr',
                           'f', 'i', 'm', 'p', 'q', 'r', 'tg', 't']
        self.res = {
            'token':None,
            'token_clean':None,
        }

    def token(self):
        startSeg = clock()
        token = list(jieba.posseg.cut(self.input.strip()))

        self.res['token'] = [x.word for x in token]
        self.res['token_clean'] = [x.word for x in filter(lambda x:x.flag not in self.stop_flags, token)]

        finishSeg = clock()
        print("分词耗时： %d 秒" % (finishSeg - startSeg))

        return self.res


if __name__ == '__main__':
    doc = '我爱北京天安门'
    t = Token(doc)
    print(t.token())

