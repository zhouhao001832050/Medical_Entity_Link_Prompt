import os
import math
import json
from random import *
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

class BM25:
    def __init__(self, corpus, tokenizer = None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except:
                    nd[word] = 1
            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)
    
    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []

        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1)) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))

        return score.tolist()


class Bm25_processor:
    def __init__(self,language="EN"):
        self.language = language
        self.tokenize_tag = " "

    def getBm25_data_format(self, input_path, mode, data_type):
        df = pd.read_excel(input_path)
        left, original_right = df["原始词"].to_list(), df["标准词"].to_list()
        right = []
        for r in original_right:
            if r not in right:
                right.append(r)

        with open(f"datasets/bm25/{data_type}/{mode}.json", "w") as f_w:
            for i, query in enumerate(left):
                right_temp = [r for r in right if r != original_right[i]]
                # print(right_temp)
                tokenized_corpus = [doc.split(self.tokenize_tag) for doc in right_temp]
                tokenized_query = query.split(self.tokenize_tag)
                if self.language == "CN":
                    tokenized_corpus = [list(doc) for doc in right_temp]
                    # print(tokenized_corpus)
                    tokenized_query = list(query)
                    # print(tokenized_query)

                bm25 = BM25Okapi(tokenized_corpus)
                negative_samples = bm25.get_top_n(query, right_temp, n=5)
                if data_type == "fewshots":
                    negative_samples = bm25.get_top_n(query, right_temp, n=1)
                all_sample = [(query, original_right[i],1)] + [(query,x,0) for x in negative_samples]
                if mode in ["dev", "test"]:
                    all_sample = [(query, original_right[i],1)]
                for element in all_sample:
                    corpus, entity, label = element[0], element[1], element[2]
                    d = {}
                    d["corpus"] = corpus
                    d["entity"] = entity
                    d["label"] = label
                    f_w.write(json.dumps(d, ensure_ascii=False) + "\n")


    def getBm25_fewshots_data_format(self, input_path, mode, data_type):
        df = pd.read_excel(input_path)
        left, original_right = df["原始词"].to_list(), df["标准词"].to_list()
        right = []
        for r in original_right:
            if r not in right:
                right.append(r)
        all = []
        with open(f"datasets/bm25/{data_type}/{mode}.json", "w") as f_w:
            for i, query in enumerate(left):
                right_temp = [r for r in right if r != original_right[i]]
                # print(right_temp)
                tokenized_corpus = [doc.split(self.tokenize_tag) for doc in right_temp]
                tokenized_query = query.split(self.tokenize_tag)
                if self.language == "CN":
                    tokenized_corpus = [list(doc) for doc in right_temp]
                    # print(tokenized_corpus)
                    tokenized_query = list(query)
                    # print(tokenized_query)

                bm25 = BM25Okapi(tokenized_corpus)
                negative_samples = bm25.get_top_n(query, right_temp, n=5)
                # if data_type == "fewshots":
                #     negative_samples = bm25.get_top_n(query, right_temp, n=1)
                all_sample = [(query, original_right[i],1)] + [(query,x,0) for x in negative_samples]
                all.append(all_sample)
                if mode in ["dev", "test"]:
                    all_sample = [(query, original_right[i],1)]
                    for element in all_sample:
                        corpus, entity, label = element[0], element[1], element[2]
                        d = {}
                        d["corpus"] = corpus
                        d["entity"] = entity
                        d["label"] = label
                        f_w.write(json.dumps(d, ensure_ascii=False) + "\n")
            train_samples = sample(all, int(len(all)/10))
            
            for train_sample in train_samples:
                for element in train_sample:
                    corpus, entity, label = element[0], element[1], element[2]
                    d = {}
                    d["corpus"] = corpus
                    d["entity"] = entity
                    d["label"] = label
                    if mode == "train":

                        f_w.write(json.dumps(d, ensure_ascii=False) + "\n")




if __name__ == "__main__":
    train_data_dir = "source_data/train.xlsx"
    dev_data_dir = "source_data/dev.xlsx"
    test_data_dir = "source_data/test.xlsx"
    processor= Bm25_processor(language="CN")
    # processor.getBm25_data_format(train_data_dir, "train", "fullshots")
    # processor.getBm25_data_format(dev_data_dir, "dev", "fullshots")
    # processor.getBm25_data_format(test_data_dir, "test", "fullshots")
    processor.getBm25_fewshots_data_format(train_data_dir, "train", "fewshots")
    # processor.getBm25_fewshots_data_format(dev_data_dir, "dev", "fewshots")
    # processor.getBm25_fewshots_data_format(test_data_dir, "test", "fewshots")
    # corpus = [
    #     "Hello there good man!",
    #     "It is quite windy in London",
    #     "How is the weather today?"
    # ]

    # tokenized_corpus = [doc.split(" ") for doc in corpus]

    # bm25 = BM25Okapi(tokenized_corpus)

    # query = "windy London"
    # tokenized_query = query.split(" ")

    # doc_scores = bm25.get_scores(tokenized_query)
    # print(bm25.get_top_n(tokenized_query, corpus, n=1))
