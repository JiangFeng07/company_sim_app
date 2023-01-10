#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/6 11:55
# @Author: lionel
import os.path
import pickle
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfModel(object):
    """tf_idf模型"""

    def __init__(self, model_path, tokenizer, file_path=None):
        """
            tf_idf模型初始化
        :param file_path: 训练语料路径
        :param model_path: 模型存储路径
        :param tokenizer: 分词器
        """
        self.file_path = file_path
        self.model_path = model_path
        self.tokenizer = tokenizer
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))

    def train(self):
        print('！！！模型开始训练！！！')
        start_time = time.time()
        """训练企业名词tf_idf模型"""
        texts = list()
        if not self.file_path or not os.path.exists(self.file_path):
            print('模型训练语料路径不存在！！！')
            sys.exit()
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = self.tokenizer.lcut(line.strip())
                texts.append(' '.join(words))
                count += 1
                if count % 1000 == 0:
                    print(count)
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(texts)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        pickle.dump(vectorizer, open(self.model_path, 'wb'))
        print('模型训练完成，耗时：%fs' % (round(time.time() - start_time, 2)))

    def get_vector(self, text):
        """预测文本tf_idf向量(稀疏表示)"""
        words = self.tokenizer.lcut(text)
        vector = self.model.transform([' '.join(words)])
        return vector


if __name__ == '__main__':
    import jieba
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--words_path', help='分词自定义词库', type=str, default='')
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='/tmp/company_name_vector_model.pkl')
    parser.add_argument('--file_path', help='模型训练语料路径', type=str, default='')
    args = parser.parse_args()
    jieba.load_userdict(args.words_path)
    model = TfIdfModel(model_path=args.model_path, tokenizer=jieba, file_path=args.file_path)
    # model.train()
    a = model.get_vector('北京大学经济管理信息学院')
