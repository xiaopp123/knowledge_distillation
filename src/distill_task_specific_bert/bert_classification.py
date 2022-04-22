# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, RandomSampler, \
    SequentialSampler, DataLoader

from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertPreTrainedModel

import numpy as np
from tqdm import tqdm, trange

from src.process import Processor
from sklearn.metrics import f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = ["[CLS]"] + tokens[:max_seq - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq - len(input_ids))
        label_id = label_map[example.label]
        features.append(InputFeatures(
            input_ids=input_ids + padding,
            input_mask=input_mask + padding,
            label_id=label_id))
    return features


def compute_metrics(preds, labels):
    return {
        'ac': (preds == labels).mean(),
        'f1': f1_score(y_true=labels, y_pred=preds)
    }


class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids):
        pooled_output = self.bert(input_ids,
                                  attention_mask=input_mask).pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels),
                            label_ids.view(-1))
        return logits


def run_bert(bert_model='bert-base-chinese', cache_dir=None,
             max_seq=128, batch_size=32, num_epochs=3, lr=2e-5):
    processor = Processor()
    train_example_list = processor.get_dev_examples('../../data/hotel')
    label_list = processor.get_labels()
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # model = BertModel.from_pretrained(bert_model)
    # outputs = model(**inputs)
    # 构建BERT模型
    model = BertClassification.from_pretrained(
        bert_model, cache_dir=cache_dir, num_labels=len(label_list))

    # print(outputs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 这些参数为什么不梯度更新
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.00
        }
    ]
    num_train_steps = int(len(train_example_list) / batch_size * num_epochs)
    # 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_features = convert_examples_to_features(
        train_example_list, label_list, max_seq, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                  dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features],
                                 dtype=torch.long)
    # 构建数据集
    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size)
    # 模型训练
    model.train()
    for _ in trange(num_epochs, desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            loss = model(input_ids, input_mask, label_ids)
            # 梯度回传
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
        print('tr_loss', tr_loss)

    # 模型验证
    print('eval...')
    eval_examples = processor.get_train_examples('../../data/hotel')
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq, tokenizer)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features],
                                  dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features],
                                   dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features],
                                  dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=batch_size)
    model.eval()
    preds = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            # 验证时梯度不更新
            logits = model(input_ids, input_mask, None)
            preds.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.vstack(preds), axis=1)
    print(compute_metrics(preds, eval_label_ids.numpy()))
    # 保存模型
    torch.save(model, '../../data/cache/model')


def main():
    run_bert()


if __name__ == '__main__':
    main()
