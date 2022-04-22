# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from src.distill_task_specific_bert.bert_classification import BertClassification


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Teacher(object):
    """ 教师网络，主要用来预估
    """
    def __init__(self, bert_model='bert-base-chinese', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        self.model = torch.load('../../data/cache/model')
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))

        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


class RNN(nn.Module):
    """ BiLSTM文本分类模型
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(0.2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.project_layer = nn.Linear(in_features=hidden_dim*2,
                                       out_features=output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, lens=None):
        embed = self.dropout(self.embedding(input))
        out, _ = self.lstm(embed)
        logits = self.project_layer(out[:, -1, :])
        return self.softmax(logits), self.log_softmax(logits), logits


def main():
    pass


if __name__ == '__main__':
    main()
