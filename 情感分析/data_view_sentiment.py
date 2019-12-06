# coding=utf-8


# 导入Style类，用于定义样式风格
from pyecharts import Style
# 导入Geo组件，用于生成地理坐标类图
from pyecharts import Geo
import json
# 导入Geo组件，用于生成柱状图
from pyecharts import Bar, Line, Overlap
# 导入Counter类，用于统计值出现的次数
from collections import Counter
import pandas as pd
from pyecharts import Pie
import fileinput,re

# 设置全局主题风格
from pyecharts import configure
configure(global_theme='wonderland')

def is_float(str):
    try:
        float(str)
    except:
        return False
    else:
        return True

def render():
    with open(r'C:\Users\think\Desktop\情感分析\doc\all.csv', mode='r', encoding='utf_8_sig') as f:
        rows = f.readlines()
        #print(rows)
        score_pos = 0
        score_neg = 0
        for row in rows[1:]:
            if row.count(',') != 1:
                continue
            elements = row.split(',')
            score = elements[1]
            if is_float(score):
                print(score)
                if float(score) > 0.5:
                    score_pos = score_pos + 1
                else:
                    score_neg = score_neg + 1
    
    print(score_pos, score_neg)

    pie = Pie("情感分析结果分布图", "数据来源：采集自猫眼",title_pos='center',width=900)
    attr = ['positive', 'negative']
    value = [score_pos, score_neg]
    print(value)
    pie.add("", attr, value, is_label_show=True, is_more_utils=True)
    pie.render(path = r'C:\Users\think\Desktop\情感分析\picture\情感分析结果分布图.html')

render()