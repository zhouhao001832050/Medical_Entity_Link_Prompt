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
            "NormLeven":self.normal_leven,
            "MiniEditDistance":self.mini_edit_distance
            
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

    # def getLongestSameStr(self, str1, str2):
    #     # 判断两个字符串长短，取短的那个进行操作
    #     if len(str1) > len(str2):
    #         str1, str2 = str2, str1

    #     # 用列表来接收最终的结果，以免出现同时有多个相同长度子串被漏查的情况
    #     resList = []

    #     # 从str1全长开始进行检测，逐渐检测到只有1位
    #     for i in range(len(str1), 0, -1):
    #         # 全长情况下不对切片进行遍历
    #         if i == len(str1):
    #             if str1 in str2:
    #                 resList.append(str1)
    #         # 非全长情况下，对str1进行切片由0到当前检测长度，迭代到str1的最后                      
    #         else:
    #             j = 0
    #             while i < len(str1):
    #                 testStr = str1[j:i]
    #                 if testStr in str2:
    #                     resList.append(testStr)
    #                 i += 1
    #                 j += 1
    #         # 判断当前长度下，是否存在子串
    #         if len(resList) > 0:
    #             return resList
    #     if len(resList) >0 :
    #         count = max(map(len, tempt))
    #     else:
    #         count = 0
    #     return count

    def getLongestSameStr(self, word1: str, word2: str) -> int:

        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # dp[i][j]代表word1以i结尾,word2以j结尾，的最大公共子串的长度

        max_len = 0
        row = 0
        col = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if max_len < dp[i][j]:
                        max_len = dp[i][j]
                        row = i
                        col = j

        max_str = ""
        i = row
        j = col
        while i > 0 and j > 0:
            if dp[i][j] == 0:
                break
            i -= 1
            j -= 1
            max_str += word1[i]

        lcstr = max_str[::-1]
        # 回溯的得到的最长公共子串
        # print(lcstr)
        return max_len


    def getDiceSimilarity(self, str1, str2):
        """dice coefficient 2nt/na + nb."""
        a_bigrams = set(str1)
        b_bigrams = set(str2)
        overlap = len(a_bigrams & b_bigrams)
        return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))


    def minDistance(self, str1, str2):
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

    def mini_edit_distance(self, str1, str2):
        dp = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
        for i in range(len(str1) + 1):
            dp[i][0] = i
        for j in range(len(str2) + 1):
            dp[0][j] = j
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        # print(dp[-1][-1])
        return dp[-1][-1]


    def get_data_format(self, input_path, mode, data_type, prosessor_name):
        df = pd.read_excel(input_path)
        left, original_right = df["原始词"].to_list(),df["标准词"].to_list()
        right = []
        for r in original_right:
            if r not in right:
                right.append(r)
        data = list(zip(left, original_right))
        length = len(data)
        with open(f"datasets/{processor_name}/{data_type}/{mode}.json", "w") as f_w:
            for i in tqdm(range(length)):
                left = data[i][0]
                right_correct = data[i][1]
                if left == "经右胸微创房间隔缺损封堵术":
                    print(right_correct)
                candidates = []
                for r in right:
                    if r != right_correct:
                        # tempt = self.getDiceSimilarity(left, r)
                        tempt = self.process_method[str(prosessor_name)](left, r)
                        candidates.append((r, tempt))
                ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:5]
                if prosessor_name == "MiniEditDistance":
                    ranked = sorted(candidates, key=lambda x: x[1], reverse = False)[:5]

                if data_type == "fewshots":
                    ranked = sorted(candidates, key=lambda x: x[1], reverse = True)[:1]
                    if prosessor_name == "MiniEditDistance":
                        ranked = sorted(candidates, key=lambda x: x[1], reverse = False)[:1]

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
    # prosessors = ["dicesimi","Minidistance","NormLeven"]
    # processors = ["LongestSamestr","Minidistance","NormLeven"]
    processors = ["dicesimi","jaccard","LongestSamestr","MiniEditDistance"]

    processor = DataProcessor(data_dir="")
    for processor_name in processors:
        processor.get_data_format(train_data_dir, "train", "fullshots", processor_name)
        processor.get_data_format(dev_data_dir, "dev", "fullshots", processor_name)
        processor.get_data_format(test_data_dir, "test", "fullshots", processor_name)
        processor.get_data_format(train_data_dir, "train", "fewshots", processor_name)
        processor.get_data_format(dev_data_dir, "dev", "fewshots", processor_name)
        processor.get_data_format(test_data_dir, "test", "fewshots", processor_name)
