#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/6 11:47
# @Author: lionel
import json
import re


class CompanyNameStruct(object):
    """企业名称结构化"""

    def __init__(self, district_path, company_dict, tokenizer):
        self.district_data = json.load(open(district_path, 'r', encoding='utf-8'))
        self.company_composition_data = json.load(open(company_dict, 'r', encoding='utf-8'))
        self.district_city_province_map = self.get_district_city_province_map()
        self.tokenizer = tokenizer

    def get_district_city_province_map(self):
        """省市区映射关系"""
        district_city_province_map = dict()
        for key, val in self.district_data['province_city_district'].items():
            province, city = tuple(key.split('_'))
            for ele in val:
                if ele not in district_city_province_map.keys():
                    district_city_province_map[ele] = set()
                district_city_province_map[ele].add((province, city))
            return district_city_province_map

    def company_name_struct(self, text):
        """企业名称结构化"""
        struct_info = {'province': '', 'city': '', 'district': '', 'brand': '', 'trade': '', 'suffix': '', 'core': ''}
        if not text:
            return struct_info
        text = re.sub('(?:经济技术开发区|高新技术产业开发区)', '', text)
        _suffix = ''
        if re.search('(?:分公司|支行|网点|办事处|代表处|分理处|服务厅|部|药房|经销点|停车区|储蓄所|农场|营业所|洗煤厂|饭庄)', text):
            suffix_parse = re.findall(
                '(.*?(?:集团|公司|站|矿|厂))(.*?(?:分公司|支行|网点|办事处|代表处|分理处|服务厅|部|药房|经销点|停车区|储蓄所|农场|营业所|洗煤厂|饭庄|分厂))$', text)
            if suffix_parse:
                text, _suffix = suffix_parse[0]
        for ele in self.company_composition_data['suffix']:
            if text.endswith(ele):
                text = re.sub('%s$' % re.escape(ele), '', text)
                _suffix = ele + _suffix
                break
        struct_info['suffix'] = re.sub('[()（）]', '', _suffix)
        words = self.tokenizer.lcut(text)
        _text = []
        brand_index = -1
        trade_index = -1
        for index, ele in enumerate(words):
            word, tag = ele
            if 'province' in tag and not struct_info['province']:
                text = text.replace(word, '')
                struct_info['province'] = self.district_data['province'].get(word, word)
            elif 'city' in tag and not struct_info['city']:
                text = text.replace(word, '')
                struct_info['city'] = self.district_data['city'].get(word, word)
            elif 'district' in tag and not struct_info['district']:
                text = text.replace(word, '')
                struct_info['district'] = self.district_data['district'].get(word, word)
            elif 'trade' in tag and re.search('%s' % word, text):
                if trade_index == -1 or index - trade_index == 1:
                    struct_info['trade'] += word
                    trade_index = index
            else:
                if brand_index == -1 or index - brand_index == 1:
                    _text.append(word)
                    brand_index = index
        struct_info['core'] = re.sub('[()（）]', '', text)  # 除去尾缀词和省市区信息部分文本
        if not struct_info['province']:
            if struct_info['city']:
                struct_info['province'] = self.district_data['city_province_map'].get(struct_info['city'],
                                                                                      struct_info['city'])
        if not struct_info['city']:
            if struct_info['district']:
                city_provinces = self.district_city_province_map.get(struct_info['district'], [])
                if len(city_provinces) == 1:
                    province, city = list(city_provinces)[0]
                    if struct_info['province'] and struct_info['province'] == province:
                        struct_info['city'] = city
                    elif not struct_info['province']:
                        struct_info['province'] = province
                        struct_info['city'] = city
                elif len(city_provinces) > 1:
                    for province, city in city_provinces:
                        if struct_info['province'] and struct_info['province'] == province:
                            struct_info['city'] = city
                            break
        struct_info['brand'] = ''.join(_text)
        return struct_info

    def get_company_compositions(self, company_name):
        """获取企业名称组成部分"""
        company_compositions = {'province': '', 'city': '', 'district': '', 'core': ''}
        compositions = self.company_name_struct(company_name)
        for k, v in company_compositions.items():
            company_compositions[k] = compositions[k]
        return company_compositions

