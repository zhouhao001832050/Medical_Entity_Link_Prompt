# -*- coding:UTF-8 -*-
import re
import sys
import math
import json
import jieba
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count



class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.process_method = {
            "dicesimi":self.getDiceSimilarity,
            "jaccard": self.getJaccardSimilarity,
            "LongestSamestr":self.getLongestSameStr,
            "Minidistance":self.minDistance,
            "NormLeven":self.normal_leven
        }

    def getJaccardSimilarity(self, str1, str2):
        terms1 = jieba.cut(str1)
        terms2 = jieba.cut(str2)
        grams1 = set(terms1)
        grams2 = set(terms2)
        temp = 0
        for i in grams1:
            if i in grams2:
                temp += 1
        demoninator = len(grams2) + len(grams1) - temp # 并集
        jaccard_coefficient = float(temp/demoninator)   # 交集
        return jaccard_coefficient

    def getLongestSameStr(self, str1, str2):
        # 判断两个字符串长短，取短的那个进行操作
        if len(str1) > len(str2):
            str1, str2 = str2, str1

        # 用列表来接收最终的结果，以免出现同时有多个相同长度子串被漏查的情况
        resList = []

        # 从str1全长开始进行检测，逐渐检测到只有1位
        for i in range(len(str1), 0, -1):
            # 全长情况下不对切片进行遍历
            if i == len(str1):
                if str1 in str2:
                    resList.append(str1)
            # 非全长情况下，对str1进行切片由0到当前检测长度，迭代到str1的最后                      
            else:
                j = 0
                while i < len(str1):
                    testStr = str1[j:i]
                    if testStr in str2:
                        resList.append(testStr)
                    i += 1
                    j += 1
            # 判断当前长度下，是否存在子串
            if len(resList) > 0:
                return resList
        if len(resList) >0 :
            count = max(map(len, tempt))
        else:
            count = 0
        return count

    def getDiceSimilarity(self, str1, str2):
            """dice coefficient 2nt/na + nb."""
            a_bigrams = set(str1)
            b_bigrams = set(str2)
            overlap = len(a_bigrams & b_bigrams)
            return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

    def minDistance(saelf, str1, str2):
        n = len(str1)
        m = len(str2)

        # 有一个字符串为空串
        if n * m == 0:
            return n + m

        # DP 数组
        D = [ [0] * (m + 1) for _ in range(n + 1)]

        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j

        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1] 
                if str1[i - 1] != str2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)

        return D[n][m]

    def normal_leven(self, str1, str2):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        #create matrix
        matrix = [0 for n in range(len_str1 * len_str2)]
        #init x axis
        for i in range(len_str1):
            matrix[i] = i
        #init y axis
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1

        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                                matrix[j*len_str1+(i-1)]+1,
                                matrix[(j-1)*len_str1+(i-1)] + cost)

        return matrix[-1]

    def get_data_format(self, input_path, mode, data_type, prosessor_name):
        df = pd.read_excel(input_path)
        left, original_right = df["原始词"].to_list(),df["标准词"].to_list()
        right = []
        for r in original_right:
            if r not in right:
                right.append(r)
        data = list(zip(left, right))
        length = len(data)
        with open(f"datasets/{processor_name}/{data_type}/{mode}.json", "w") as f_w:
            for i in tqdm(range(length)):
                left = data[i][0]
                right_correct = data[i][1]
                candidates = []
                for r in right:
                    if r != right_correct:
                        # tempt = self.getDiceSimilarity(left, r)
                        tempt = self.process_method[str(prosessor_name)](left, r)
                        candidates.append((r, tempt))
                ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
                if data_type == "fewshots":
                    ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
                all_sample = [(left, data[i][1],1)] + [(left,x[0],0) for x in ranked]
                if mode in ["dev", "test"]:
                    all_sample = [(left, data[i][1],1)]
                for element in all_sample:
                    corpus, entity, label = element[0], element[1], element[2]
                    d = {}
                    d["corpus"] = corpus
                    d["entity"] = entity
                    d["label"] = label
                    f_w.write(json.dumps(d, ensure_ascii=False) + "\n")


# class JaccardProcessor:
#     """jaccard score processor"""
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
  

#     def getJaccardSimilarity(self, str1, str2):
#         terms1 = jieba.cut(str1)
#         terms2 = jieba.cut(str2)
#         grams1 = set(terms1)
#         grams2 = set(terms2)
#         temp = 0
#         for i in grams1:
#             if i in grams2:
#                 temp += 1
#         demoninator = len(grams2) + len(grams1) - temp # 并集
#         jaccard_coefficient = float(temp/demoninator)   # 交集
#         return jaccard_coefficient


#     def get_data_format(self, input_path, mode, data_type):
#         df = pd.read_excel(input_path)
#         left, right = df["原始词"].to_list(),df["标准词"].to_list()
#         data = list(zip(left, right))
#         length = len(data)
#         negative = []
#         for i in tqdm(range(length)):
#             if mode in ["dev", "test"]:
#                 continue
#             left = data[i][0]
#             candidates = []
#             for j in range(length):
#                 if j != i:
#                     right = data[j][1]
#                     tempt = self.getJaccardSimilarity(left, right)
#                     candidates.append((right, tempt))
#             ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
#             if data_type == "fewshots":
#                 ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[0]
#                 print(ranked)
#             negative.append((left, ranked[0][0]))
#         positive = [(x[0], x[1], 1) for x in data]
#         negative = [(x[0], x[1], 0) for x in negative]
#         res = (positive + negative)
#         if mode in ["dev", "test"]:
#             res = positive
#         random.shuffle(res)
#         all_data = []
#         with open(f"datasets/jaccard/{data_type}/{mode}.json", "w") as f_w:
#             for element in res:
#                 d = {}
#                 corpus, entity, label = element[0], element[1], element[2]
#                 d["corpus"] = corpus
#                 d["entity"] = entity
#                 d["label"] = label
#                 f_w.write(json.dumps(d, ensure_ascii=False) + "\n")


# class LongestSameStrProcessor:
#     """longest score processor"""
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
    
#     def getLongestSameStr(self, str1, str2):
#         # 判断两个字符串长短，取短的那个进行操作
#         if len(str1) > len(str2):
#             str1, str2 = str2, str1

#         # 用列表来接收最终的结果，以免出现同时有多个相同长度子串被漏查的情况
#         resList = []

#         # 从str1全长开始进行检测，逐渐检测到只有1位
#         for i in range(len(str1), 0, -1):
#             # 全长情况下不对切片进行遍历
#             if i == len(str1):
#                 if str1 in str2:
#                     resList.append(str1)
#             # 非全长情况下，对str1进行切片由0到当前检测长度，迭代到str1的最后                      
#             else:
#                 j = 0
#                 while i < len(str1):
#                     testStr = str1[j:i]
#                     if testStr in str2:
#                         resList.append(testStr)
#                     i += 1
#                     j += 1
#             # 判断当前长度下，是否存在子串
#             if len(resList) > 0:
#                 return resList
#         return resList

#     def get_data_format(self, input_path, mode,data_type):
#         df = pd.read_excel(input_path)
#         left, right = df["原始词"].to_list(),df["标准词"].to_list()
#         data = list(zip(left, right))

#         length = len(data)
#         negative = []
#         for i in tqdm(range(length)):
#             if mode in ["dev", "test"]:
#                 continue
#             left = data[i][0]
#             candidates = []
#             for j in range(length):
#                 if j != i:
#                     right = data[j][1]
#                     tempt = self.getLongestSameStr(left, right)
#                     if tempt:
#                         count = max(map(len, tempt))
#                     else:
#                         count = 0
#                     candidates.append((right, count))
#             ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
            
#             if data_type == "fewshots":
#                 ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
#             negative.append((left, ranked[0][0]))
#         positive = [(x[0], x[1], 1) for x in data]
#         negative = [(x[0], x[1], 0) for x in negative]
#         res = (positive + negative)
#         if mode in ["dev", "test"]:
#             res = positive
#         random.shuffle(res)
#         all_data = []
#         with open(f"datasets/LongestSameStr/{data_type}/{mode}.json", "w") as f_w:
#             for element in res:
#                 d = {}
#                 corpus, entity, label = element[0], element[1], element[2]
#                 d["corpus"] = corpus
#                 d["entity"] = entity
#                 d["label"] = label
#                 f_w.write(json.dumps(d, ensure_ascii=False) + "\n")



# class DiceSimilarityProcessor:
#     """dice coefficient 2nt/na + nb"""
#     def __init__(self, data_dir):
#         self.data_dir = data_dir

#     def getDiceSimilarity(self, str1, str2):
#         """dice coefficient 2nt/na + nb."""
#         a_bigrams = set(str1)
#         b_bigrams = set(str2)
#         overlap = len(a_bigrams & b_bigrams)
#         return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

#     def get_data_format(self, input_path, mode, data_type):
#         df = pd.read_excel(input_path)
#         left, original_right = df["原始词"].to_list(),df["标准词"].to_list()
#         right = []
#         for r in original_right:
#             if r not in right:
#                 right.append(r)
#         data = list(zip(left, right))
#         length = len(data)
#         with open(f"datasets/dicesimi/{data_type}/{mode}.json", "w") as f_w:
#             for i in tqdm(range(length)):
#                 print(data[i])
#                 left = data[i][0]
#                 right_correct = data[i][1]

#                 candidates = []

#                 for r in right:
#                     if r != right_correct:
#                         tempt = self.getDiceSimilarity(left, r)
#                         candidates.append((r, tempt))
#                 ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
#                 if data_type == "fewshots":
#                     ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
#                 all_sample = [(left, data[i][1],1)] + [(left,x[0],0) for x in ranked]
#                 if mode in ["dev", "test"]:
#                     all_sample = [(left, data[i][1],1)]
#                 for element in all_sample:
#                     corpus, entity, label = element[0], element[1], element[2]
#                     d = {}
#                     d["corpus"] = corpus
#                     d["entity"] = entity
#                     d["label"] = label
#                     f_w.write(json.dumps(d, ensure_ascii=False) + "\n")



# class MinDistanceProcessor:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir

#     def minDistance(saelf, word1, word2):
#         n = len(word1)
#         m = len(word2)

#         # 有一个字符串为空串
#         if n * m == 0:
#             return n + m

#         # DP 数组
#         D = [ [0] * (m + 1) for _ in range(n + 1)]

#         # 边界状态初始化
#         for i in range(n + 1):
#             D[i][0] = i
#         for j in range(m + 1):
#             D[0][j] = j

#         # 计算所有 DP 值
#         for i in range(1, n + 1):
#             for j in range(1, m + 1):
#                 left = D[i - 1][j] + 1
#                 down = D[i][j - 1] + 1
#                 left_down = D[i - 1][j - 1] 
#                 if word1[i - 1] != word2[j - 1]:
#                     left_down += 1
#                 D[i][j] = min(left, down, left_down)

#         return D[n][m]


    # def get_data_format(self, input_path, mode,data_type, processor_type):
    #     df = pd.read_excel(input_path)
    #     left, right = df["原始词"].to_list(),df["标准词"].to_list()
    #     data = list(zip(left, right))
    #     length = len(data)
    #     negative = []
    #     for i in tqdm(range(length)):
    #         if mode in ["dev", "test"]:
    #             continue
    #         left = data[i][0]
    #         candidates = []
    #         for j in range(length):
    #             if j != i:
    #                 right = data[j][1]
    #                 tempt = self.minDistance(left, right)
    #                 candidates.append((right, tempt))
    #         ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
    #         if data_type == "fewshots":
    #             ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
    #         negative.append((left, ranked[0][0]))
    #     positive = [(x[0], x[1], 1) for x in data]
    #     negative = [(x[0], x[1], 0) for x in negative]

    #     res = (positive + negative)
    #     if mode in ["dev", "test"]:
    #         res = positive
    #     random.shuffle(res)
    #     all_data = []
    #     with open(f"datasets/Minidistance/{data_type}/{mode}.json", "w") as f_w:
    #         for element in res:
    #             d = {}
    #             corpus, entity, label = element[0], element[1], element[2]
    #             d["corpus"] = corpus
    #             d["entity"] = entity
    #             d["label"] = label
    #             f_w.write(json.dumps(d, ensure_ascii=False) + "\n")




# class NormalLevenProcessor:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir

#     def normal_leven(self, str1, str2):
#         len_str1 = len(str1) + 1
#         len_str2 = len(str2) + 1
#         #create matrix
#         matrix = [0 for n in range(len_str1 * len_str2)]
#         #init x axis
#         for i in range(len_str1):
#             matrix[i] = i
#         #init y axis
#         for j in range(0, len(matrix), len_str1):
#             if j % len_str1 == 0:
#                 matrix[j] = j // len_str1

#         for i in range(1, len_str1):
#             for j in range(1, len_str2):
#                 if str1[i-1] == str2[j-1]:
#                     cost = 0
#                 else:
#                     cost = 1
#                 matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
#                                 matrix[j*len_str1+(i-1)]+1,
#                                 matrix[(j-1)*len_str1+(i-1)] + cost)

#         return matrix[-1]


#     def get_data_format(self, input_path, mode,data_type):
#         df = pd.read_excel(input_path)
#         left, right = df["原始词"].to_list(),df["标准词"].to_list()
#         data = list(zip(left, right))
#         length = len(data)
#         negative = []
#         for i in tqdm(range(length)):
#             if mode in ["dev", "test"]:
#                 continue
#             left = data[i][0]
#             candidates = []
#             for j in range(length):
#                 if j != i:
#                     right = data[j][1]
#                     tempt = self.normal_leven(left, right)
#                     candidates.append((right, tempt))
#             ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
#             if data_type == "fewshots":
#                 ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
#             negative.append((left, ranked[0][0]))
#         positive = [(x[0], x[1], 1) for x in data]
#         negative = [(x[0], x[1], 0) for x in negative]

#         res = (positive + negative)
#         if mode in ["dev", "test"]:
#             res = positive
#         random.shuffle(res)
#         all_data = []
#         with open(f"datasets/NormLeven/{mode}.json", "w") as f_w:
#             for element in res:
#                 d = {}
#                 corpus, entity, label = element[0], element[1], element[2]
#                 d["corpus"] = corpus
#                 d["entity"] = entity
#                 d["label"] = label
#                 f_w.write(json.dumps(d, ensure_ascii=False) + "\n")

    




if __name__ == "__main__":
    train_data_dir = "source_data/train.xlsx"
    dev_data_dir = "source_data/dev.xlsx"
    test_data_dir = "source_data/test.xlsx"

    # "dicesimi":self.getDiceSimilarity(str1, str2),
    #         "jaccard": self.getJaccardSimilarity(str1, str2),
    #         "LongestSamestr":self.getLongestSameStr(str1, str2),
    #         "Minidistance":self.minDistance(str1, str2),
    #         "NormLeven":self.normal_leven(str1, str2)
    # prosessors = ["dicesimi", "jaccard","LongestSamestr","Minidistance","NormLeven"]
    # prosessors = ["jaccard"]
    prosessors = ["dicesimi","Minidistance","NormLeven"]

    processor = DataProcessor(data_dir="")
    for processor_name in prosessors:
        processor.get_data_format(train_data_dir, "train", "fullshots", processor_name)
        processor.get_data_format(dev_data_dir, "dev", "fullshots", processor_name)
        processor.get_data_format(test_data_dir, "test", "fullshots", processor_name)
        processor.get_data_format(train_data_dir, "train", "fewshots", processor_name)
        processor.get_data_format(dev_data_dir, "dev", "fewshots", processor_name)
        processor.get_data_format(test_data_dir, "test", "fewshots", processor_name)

    # processor= LongestSameStrProcessor(data_dir="")
    # processor.get_data_format(train_data_dir, "train", "fullshots")
    # processor.get_data_format(dev_data_dir, "dev", "fullshots")
    # processor.get_data_format(test_data_dir, "test", "fullshots")
    # processor.get_data_format(train_data_dir, "train", "fewshots")
    # processor.get_data_format(dev_data_dir, "dev", "fewshots")
    # processor.get_data_format(test_data_dir, "test", "fewshots")
    # print("======================================")
    # processor= DiceSimilarityProcessor(data_dir="")
    # processor.get_data_format(train_data_dir, "train", "fullshots")
    # # processor.get_data_format(dev_data_dir, "dev", "fullshots")
    # # processor.get_data_format(test_data_dir, "test", "fullshots")
    # processor.get_data_format(train_data_dir, "train", "fewshots")
    # processor.get_data_format(dev_data_dir, "dev", "fewshots")
    # processor.get_data_format(test_data_dir, "test", "fewshots")
    # print("======================================")
    # processor= MinDistanceProcessor(data_dir="")
    # processor.get_data_format(train_data_dir, "train", "fullshots")
    # processor.get_data_format(dev_data_dir, "dev", "fullshots")
    # processor.get_data_format(test_data_dir, "test", "fullshots")
    # processor.get_data_format(train_data_dir, "train", "fewshots")
    # processor.get_data_format(dev_data_dir, "dev", "fewshots")
    # processor.get_data_format(test_data_dir, "test", "fewshots")
    # print("======================================")
    # processor= NormalLevenProcessor(data_dir="")
    # processor.get_data_format(train_data_dir, "train", "fullshots")
    # processor.get_data_format(dev_data_dir, "dev", "fullshots")
    # processor.get_data_format(test_data_dir, "test", "fullshots")
    # processor.get_data_format(train_data_dir, "train", "fewshots")
    # processor.get_data_format(dev_data_dir, "dev", "fewshots")
    # processor.get_data_format(test_data_dir, "test", "fewshots")

    # print("======================================")


    # processor = JaccardProcessor(data_dir="")
    # processor.get_data_format(train_data_dir, "train", "fullshots")
    # processor.get_data_format(dev_data_dir, "dev", "fullshots")
    # processor.get_data_format(test_data_dir, "test", "fullshots")
    # processor.get_data_format(train_data_dir, "train", "fewshots")
    # processor.get_data_format(dev_data_dir, "dev", "fewshots")
    # processor.get_data_format(test_data_dir, "test", "fewshots")