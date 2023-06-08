#! -*- coding:utf-8 -*-
import random
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from common import seed_everything
from tokenizers import Tokenizer

maxlen = 256
batch_size = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'train'  # train/inference
dict_path = ""

seed_everything(42)


# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D
