# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, RandomSampler, \
    SequentialSampler, DataLoader

from src.process import Processor
from src.utils import WordEmbedding, process_example
from src.distill_task_specific_bert.model import RNN, Teacher
from src.distill_task_specific_bert.bert_classification import BertClassification

import argparse
import numpy as np
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--distill', type=int, help='是否使用distill')
args = parser.parse_args()

LTensor = torch.LongTensor
FTensor = torch.FloatTensor


def teacher_predict(teacher, example_list):
    with torch.no_grad():
        res = [teacher.predict(example.text)
               for example in tqdm(example_list)]
        return np.vstack(res)


def main():

    num_epochs = 15
    batch_size = 64
    alpha = 0.0
    # 构建词表
    w2v = WordEmbedding(file_name='../../data/word2vec')
    # 使用bert的语料处理方式
    processor = Processor()
    # 加载数据
    # train_example_list = processor.get_train_examples('../data/hotel')
    # dev_example_list = processor.get_dev_examples('../data/hotel')
    train_example_list = processor.get_dev_examples('../../data/hotel')
    dev_example_list = processor.get_train_examples('../../data/hotel')
    test_example_list = processor.get_test_examples('../../data/hotel')
    train_token_list, train_label_list, train_len_list = \
        process_example(train_example_list, w2v, max_length=50)
    dev_token_list, dev_label_list, dev_len_list = \
        process_example(dev_example_list, w2v, max_length=50)
    test_token_list, test_label_list, test_len_list = \
        process_example(test_example_list, w2v)

    if args.distill == 1:
        # 加载bert模型作为Teacher
        teacher = Teacher()
        print(teacher.predict('还不错，这个价位算是物有所值'))
        # 使用bert预估得到概率分布
        teacher_pred = teacher_predict(teacher, train_example_list)

    # 构建BiLSTM模型
    model = RNN(vocab_size=w2v.vocab_size,
                emb_dim=64,
                hidden_dim=128,
                output_dim=2)
    # 交叉熵
    cross_entropy_loss = nn.NLLLoss()
    # 均方误差
    mse_loss = nn.MSELoss()
    # 优化器，lr影响较大
    opt = optim.Adam(model.parameters(), lr=0.0001)
    for _ in trange(num_epochs, desc='Epoch'):
        model.train()
        for idx in tqdm(range(0, len(train_token_list), batch_size)):
            model.zero_grad()
            # 输入文本序列
            x_list = LTensor(train_token_list[idx: idx+batch_size])
            # 真实label
            y_list = LTensor(train_label_list[idx: idx+batch_size])
            # 使用BiLSTM对文本建模，得到softmax后值和log(softmax)值
            softmax_val, log_softmax_val, logits = model(x_list)
            if args.distill == 1:
                # bert预估的概率分布
                teacher_pred_batch = FTensor(teacher_pred[idx: idx+batch_size])
                # 交叉熵和均方误差
                loss = alpha * cross_entropy_loss(log_softmax_val, y_list) + \
                       (1 - alpha) * mse_loss(softmax_val, teacher_pred_batch)
            else:
                loss = cross_entropy_loss(log_softmax_val, y_list)
            loss.backward()
            opt.step()
        # 预估
        accu = []
        with torch.no_grad():
            for idx in tqdm(range(0, len(dev_token_list), batch_size)):
                x_list = LTensor(dev_token_list[idx: idx+batch_size])
                y_list = LTensor(dev_label_list[idx: idx+batch_size])
                _, py = torch.max(model(x_list)[1], 1)
                accu.append((py == y_list).float().mean().item())
        print(np.mean(accu))


if __name__ == '__main__':
    main()
