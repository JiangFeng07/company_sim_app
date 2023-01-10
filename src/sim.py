#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/6 11:42
# @Author: lionel
import logging
import os.path
import pickle
import re
import time

import jieba
import numpy as np
from sparse_dot_topn import awesome_cossim_topn

from src.text_struct import CompanyNameStruct

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：￥＄º"
digit2china = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}


def company_name_process(company_name):
    """企业名称预处理"""
    # 去掉空格
    company_name = re.sub('[ ]+', '', company_name)

    # 去掉尾部括号的内容
    if len(re.findall('(.*)\(.*\)$', company_name)) > 0:
        company_name = re.findall('(.*)\(.*\)$', company_name)[0]
    if len(re.findall('(.*)（.*）$', company_name)) > 0:
        company_name = re.findall('(.*)（.*）$', company_name)[0]

    # 去掉标点符号
    company_name = re.sub('[%s]' % PUNCTUATION, '', company_name)

    # 小写转大写
    company_name = company_name.upper()

    # 阿拉伯数字转换为中文数字
    for k, v in digit2china.items():
        company_name = re.sub(k, v, company_name)

    return company_name


def generate_batch(file_path, batch_size=10):
    """批量迭代获取数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        line = f.readline()
        company_list = list()
        company_words = list()
        while line:
            if len(company_words) == batch_size:
                yield company_list, company_words
                company_words = list()
                company_list = list()
            company_name = company_name_process(line.strip())

            _company_words = ' '.join(jieba.lcut(company_name))
            company_words.append(_company_words)
            company_list.append(line.strip())
            line = f.readline()

        if len(company_words) != 0:
            yield company_list, company_words


def judge_company_similarity(company_compositions_a, company_compositions_b):
    """精确判断企业名称相似性"""
    if not company_compositions_a or not company_compositions_b:
        return False

    if company_compositions_a['province'] and company_compositions_b['province'] and \
            company_compositions_a['province'] != company_compositions_b['province']:
        return False

    if company_compositions_a['city'] and company_compositions_b['city'] and \
            company_compositions_a['city'] != company_compositions_b['city']:
        return False

    if company_compositions_a['district'] and company_compositions_b['district'] and \
            company_compositions_a['district'] != company_compositions_b['district']:
        return False

    if company_compositions_a['core'] and company_compositions_b['core'] and \
            company_compositions_a['core'] != company_compositions_b['core']:
        return False

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', help='训练语料、词库、模型存储根目录', type=str, default='/tmp/com_sim_data')
    parser.add_argument('--sim_threshold', help='文本相似阈值，低于阈值的不保存', type=float, default=0.3)
    parser.add_argument('--batch_size', help='批量处理大小', type=int, default=10)
    args = parser.parse_args()
    # 加载企业名称组成自定义词库
    words_path = os.path.join(args.base_path, 'company_name.dict')  # 分词自定义词库
    jieba.load_userdict(words_path)
    import jieba.posseg as psg

    industry_company_path = os.path.join(args.base_path, 'company.csv')  # 企业数据集一
    patent_company_path = os.path.join(args.base_path, 'company2.csv')  # 企业数据集二
    file_path = os.path.join(args.base_path, 'com_sim.csv')  # 结果文件
    tfidf_model_path = os.path.join(args.base_path, 'company_name_vector_model.pkl')  # 文本语义表示模型
    district_file_path = os.path.join(args.base_path, 'province_city_district.json')  # 行政区划映射文件路径
    company_file_path = os.path.join(args.base_path, 'company_name.json')  # 企业名称组成词库

    cns = CompanyNameStruct(district_file_path, company_file_path, tokenizer=psg)

    logging.info('开始加载tfidf模型。')
    vector_model = pickle.load(open(tfidf_model_path, 'rb'))
    logging.info('tfidf模型加载结束。')

    logging.info('开始企业名称匹配任务。')
    company_composition_dict = dict()
    start_time = time.time()
    result_file = open(file_path, 'w', encoding='utf-8')
    industry_company_data = generate_batch(industry_company_path, batch_size=args.batch_size)
    industry_company_next = next(industry_company_data)
    match_count = 0
    while industry_company_next:
        industry_company_dict = dict()
        for industry_company in industry_company_next[0]:
            industry_company_dict[len(industry_company_dict)] = industry_company
        industry_company_matrix = vector_model.transform(industry_company_next[1])

        patent_company_data = generate_batch(patent_company_path, batch_size=args.batch_size)
        patent_company_next = next(patent_company_data)

        while patent_company_next:
            patent_company_dict = dict()
            for patent_company in patent_company_next[0]:
                patent_company_dict[len(patent_company_dict)] = patent_company

            patent_company_matrix = vector_model.transform(patent_company_next[1])

            matches = awesome_cossim_topn(industry_company_matrix, patent_company_matrix.transpose(), 10,
                                          lower_bound=args.sim_threshold).toarray()
            row, column = np.nonzero(matches)
            for i in range(len(row)):
                patent_company = patent_company_dict[column[i]]
                industry_company = industry_company_dict[row[i]]
                if patent_company in company_composition_dict.keys():
                    patent_company_compositions = company_composition_dict[patent_company]
                else:
                    patent_company_compositions = cns.get_company_compositions(patent_company)
                    company_composition_dict[patent_company] = patent_company_compositions
                if industry_company in company_composition_dict.keys():
                    industry_company_compositions = company_composition_dict[industry_company]
                else:
                    industry_company_compositions = cns.get_company_compositions(industry_company)
                    company_composition_dict[industry_company] = industry_company_compositions
                # 排除确定不相似数据
                if judge_company_similarity(patent_company_compositions, industry_company_compositions):
                    match_count += 1
                    result_file.write(('%s\t%s\t%.2f\n' % (
                        patent_company_dict[column[i]], industry_company_dict[row[i]], matches[row[i]][column[i]])))
                    if match_count % 10000 == 0:
                        logging.info('已匹配相似企业数：%d' % match_count)
            try:
                patent_company_next = next(patent_company_data)
            except:
                break

        try:
            industry_company_next = next(industry_company_data)
        except:
            break
    result_file.close()
    logging.info('企业名称匹配任务结束，共耗时%f小时' % round((time.time() - start_time) / 3600), 2)
