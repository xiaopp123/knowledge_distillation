# -*- coding: utf-8 -*-


import jieba
# from keras.preprocessing.text import Tokenizer


class WordEmbedding(object):
    def __init__(self, file_name):
        self.word2id = {}
        self.id2word = {}
        self.w2v = {}
        self.dim = None
        self.vocab_size = None
        self.load_w2v(file_name)
        self.add_hotel_word()

    def load_w2v(self, file_name):
        idx = 2
        with open(file_name, 'r') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                line = line.split(' ')
                word, embedding = line[0], line[1:]
                if word in self.word2id:
                    continue
                self.word2id[word] = idx
                self.w2v[idx] = [float(t) for t in embedding]
                idx += 1
        self.dim = len(self.w2v[2])
        self.word2id['[PAD]'] = 0
        self.w2v[0] = [0] * self.dim
        self.word2id['OOV'] = 1
        self.w2v[1] = [0] * self.dim
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.id2word)
        print(idx)

    def add_hotel_word(self):
        with open('../../data/hotel/hotel.txt', 'r') as fr:
            idx = self.vocab_size
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                for token in jieba.cut(line.split('\t', 1)[1]):
                    if token in self.word2id:
                        continue
                    self.word2id[token] = idx
                    idx += 1
            self.id2word = {v: k for k, v in self.word2id.items()}
            self.vocab_size = len(self.id2word)

    def get_token_id(self, token):
        return self.word2id.get(token, 0)


def process_example(example_list, w2v, max_length=50):
    token_list = []
    label_list = []
    length_list = []
    for i, example in enumerate(example_list):
        text = example.text
        line_token_list = [w2v.get_token_id(t) for t in jieba.cut(text)]
        token_length = len(line_token_list)
        if token_length >= max_length:
            # 截断
            line_token_list = line_token_list[:max_length]
        else:
            # 填充
            line_token_list += [0] * (max_length - token_length)
        length_list.append(len(line_token_list))
        token_list.append(line_token_list)
        label_list.append(int(example.label))
    return token_list, label_list, length_list


def load_data(file_name, mode_list):
    w2v = WordEmbedding(file_name='../../data/word2vec')
    text_list = []
    token_list = []
    label_list = []
    with open(file_name, 'r') as fr:
        for line in fr:
            label, text = line.strip().split('\t', 1)
            text_list.append(text)
            token_list.append([w2v.get_token_id(token) for token in jieba.cut(text)])
            label_list.append(label)



def main():
    w2v = WordEmbedding(file_name='../data/word2vec')
    print(w2v.dim, w2v.vocab_size)
    # load_data('../data/hotel/train.txt')


if __name__ == '__main__':
    main()
