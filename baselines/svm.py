import os
import codecs
import jieba

base_dir = '/Users/zhouhao/Documents/Medical_Entity_Link_Prompt/datasets/MiniEditDistance/'

train_file = dataset + 'fewshots/train.json'                                # 训练集
dev_file = dataset + 'fewshots/dev.json'                                    # 验证集
test_file = dataset + 'fewshots/test.json'                                  # 测试集
vocab_path = dataset + "SVM" + '/vocab.pkl'                                 # 词表


