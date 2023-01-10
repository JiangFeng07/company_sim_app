# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/10 00:57
# @Author: lionel
import json

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, json_tuple
from pyspark.sql.types import ArrayType, StringType
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn


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


@udf(returnType=ArrayType(StringType()))
def sim(company_names_a, company_names_b, features_a, indices_a, indptr_a, compositions_a,
        shape_a, features_b, indices_b, indptr_b, compositions_b, shape_b):
    results = []
    matrix_a = csr_matrix((np.asarray(features_a), np.asarray(indices_a), np.asarray(indptr_a)),
                          shape=(shape_a[0], shape_a[1]))
    matrix_b = csr_matrix((np.asarray(features_b), np.asarray(indices_b), np.asarray(indptr_b)),
                          shape=(shape_b[0], shape_b[1]))
    matches = awesome_cossim_topn(matrix_a, matrix_b.transpose(), ntop=10, lower_bound=0.3)
    index = 0
    company_name_a_dict = dict()
    i = 0
    for company_name_a, composition_a in zip(company_names_a, compositions_a):
        company_name_a_dict[i] = {'name': company_name_a, 'composition': json.loads(composition_a)}
        i += 1
    company_name_b_dict = dict()
    i = 0
    for company_name_b, composition_b in zip(company_names_b, compositions_b):
        company_name_b_dict[i] = {'name': company_name_b, 'composition': json.loads(composition_b)}
        i += 1

    sim_list = matches.data.tolist()
    for row, column in zip(*np.nonzero(matches)):
        sim = sim_list[index]
        index += 1
        company_name_a, company_composition_a = company_name_a_dict[row].values()
        company_name_b, company_composition_b = company_name_b_dict[column].values()
        if not judge_company_similarity(company_composition_a, company_composition_b):
            continue
        results.append(json.dumps({'patent_company': company_name_a, 'industry_company': company_name_b,
                                   'sim': round(sim, 6)}, ensure_ascii=False))
    return results


if __name__ == '__main__':
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true")
    spark.conf.set("spark.blacklist.enabled", "false")
    industry_df = spark.table('nlp_industry_vector_info')
    company_df = spark.table('nlp_company_vector_info')

    df = spark.sql("""
        select
            a.company_names company_names_a, a.features features_a, a.indices indices_a, a.indptr indptr_a,
            a.shape shape_a, a.compositions compositions_a, b.company_names company_names_b, b.features features_b,
             b.indices indices_b, b.indptr indptr_b, b.shape shape_b, b.compositions compositions_b
        from nlp_company_vector_info a 
        cross join nlp_industry_vector_info b 
        where size(a.company_names)>0 and size(b.company_names)>0
    """)

    df = df.select(explode(
        sim('company_names_a', 'company_names_b', 'features_a', 'indices_a', 'indptr_a', 'compositions_a', 'shape_a',
            'features_b', 'indices_b', 'indptr_b', 'compositions_b', 'shape_b')).alias('info'))

    df = df.select(
        json_tuple('info', 'patent_company', 'industry_company', 'sim').alias('patent_company', 'industry_company',
                                                                              'sim'))
    df.write.mode('overwrite').saveAsTable('nlp_company_sim_info')
