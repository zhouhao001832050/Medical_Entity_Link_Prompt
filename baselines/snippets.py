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


