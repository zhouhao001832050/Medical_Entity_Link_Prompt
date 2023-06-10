import unicodedata
import re
import numpy as np
from torch.utils.data import Dataset
basestring=str

def is_string(s):
    """判断是否是字符串"""
    return isinstance(s, basestring)
    
class ListDataset(Dataset):
    """
    param file_path: str, file path wait to be load, None if nonexist
    param data: List[Any], list format data
    """
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, tuple, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        return self.data[index]
    

    @staticmethod
    def load_data(file_path):
        return file_path

def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen"""
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences

def lowercase_and_normalize(text, never_split=()):
    """转小写，并进行简单的标准化"""
    # if is_py2:
    #     text = unicode(text)
    
    # convert non-special tokens to lowercase
    escaped_special_toks = [re.escape(s_tok) for s_tok in never_split]
    pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
    text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    # text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text

def text_segmentate(text, maxlen, seps='\n', strips=None, truncate=True):
    """将文本按照标点符号划分为若干个短句
       
       :param text: 待划分的句子
       :param maxlen: int, 截断长度
       :param seps: 分隔符
       :param strips: ''.strip()
       :param truncate: True表示标点符号切分后仍然超长时, 按照maxlen硬截断分成若干个短句
       :return: List[str], 划分后的句子列表
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
        return texts
    elif truncate and (not seps) and (len(text) > maxlen):
        # 标点符号用完，仍然超长，且设置了truncate=True
        return [text[i*maxlen:(i+1)*maxlen] for i in range(0, int(np.ceil(len(text)/maxlen)))]
    else:
        return [text]

