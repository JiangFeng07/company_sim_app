#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/5 14:42
# @Author: lionel
import json
import pickle
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, split, explode
from pyspark.sql.types import StringType, ArrayType, StructField, StructType, IntegerType, FloatType

from src.text_struct import CompanyNameStruct

punctuations = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：￥＄º"
digit2china = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}


def company_name_process(company_name):
    """公司名称预处理"""
    # 去掉空格
    if not company_name:
        return ''
    company_name = re.sub('[ ]+', '', company_name)

    # 去掉尾部括号的内容
    if len(re.findall('(.*)\(.*\)$', company_name)) > 0:
        company_name = re.findall('(.*)\(.*\)$', company_name)[0]
    if len(re.findall('(.*)（.*）$', company_name)) > 0:
        company_name = re.findall('(.*)（.*）$', company_name)[0]

    # 去掉标点符号
    company_name = re.sub('[%s]' % punctuations, '', company_name)

    # 小写转大写
    company_name = company_name.upper()

    # 阿拉伯数字转换为中文数字
    for k, v in digit2china.items():
        company_name = re.sub(k, v, company_name)

    return company_name


def partition_parse(partition):
    import jieba.analyse
    word_path = 'vocab/company_name.dict'
    jieba.load_userdict(word_path)
    vector_model = pickle.load(open('company_name_vector_model.pkl', 'rb'))
    import jieba.posseg as psg
    cns = CompanyNameStruct('province_city_district.json', 'company_name.json', tokenizer=psg)

    def next_batch():
        company_names, company_words = [], []
        while len(company_names) < 1000:
            try:
                row = next(partition)
                company_names.append(row['company_name'])
                _company_name = company_name_process(row['company_name'])
                words = jieba.lcut(_company_name)
                company_words.append(' '.join(words))
            except StopIteration:
                break
        return company_names, company_words

    company_names, company_words = next_batch()

    while len(company_names) > 0:
        a = vector_model.transform(company_words)
        features = a.data.tolist()
        indices = a.indices.tolist()
        indptr = a.indptr.tolist()
        _company_names = company_names
        company_compositions = []
        for company_name in company_names:
            compositions = cns.get_company_compositions(company_name)
            company_compositions.append(json.dumps(compositions, ensure_ascii=False))
        company_names, company_words = next_batch()

        yield _company_names, features, indices, indptr, company_compositions, list(a.shape)


if __name__ == '__main__':
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true")

    industry_df = spark.table('industry_company_info'). \
        select(regexp_replace('company_name', ' ', '').alias('company_name')).filter(
        'length(company_name)>0').distinct()
    industry_df.persist()

    rdd = industry_df.repartition(500).rdd.mapPartitions(partition_parse)
    schema = StructType([StructField('company_names', ArrayType(StringType())),
                         StructField('features', ArrayType(FloatType())),
                         StructField('indices', ArrayType(IntegerType())),
                         StructField('indptr', ArrayType(IntegerType())),
                         StructField('compositions', ArrayType(StringType())),
                         StructField('shape', ArrayType(IntegerType()))])
    spark.createDataFrame(rdd, schema).write.format('orc').mode('overwrite').saveAsTable('nlp_industry_vector_info')

    company_df = spark.table('patent_base_info').filter("chr_assigneetype in ('企业','大专院校','科研单位','机关团体')"). \
        select(split('chr_assignee', '[;；]').alias('companies'))
    company_df = company_df.select(explode('companies').alias('company_name')).filter(
        'length(company_name)>0').distinct()
    company_df.persist()

    rdd = company_df.repartition(500).rdd.mapPartitions(partition_parse)
    spark.createDataFrame(rdd, schema).write.format('orc').mode('overwrite').saveAsTable('nlp_company_vector_info')
