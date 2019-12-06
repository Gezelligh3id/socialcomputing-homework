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

# 数据可视化

# 存放分值
scores = []
# 存放性别
genders = []

dates = []
# 情感分级结果
sentiments = []
# 正向情感指数
positive_probs = []
negative_probs = []

positive_text = ""
negative_text = ""
cities =[]
def is_float(str):
    try:
        float(str)
    except:
        return False
    else:
        return True



def render():
    # 获取评论中所有城市
    with open(r'C:\Users\think\Desktop\情感分析\doc\maoyan.csv', mode='r', encoding='utf_8_sig') as f:
        rows = f.readlines()
        #print(rows)
        #num = 0
        for row in rows[1:]:
            #print(row)
            #num = num + 1
            #print(num)
            '''
            if num == 10:
             break
            '''
            #print(row)
            #print(row.count(','))
            if row.count(',') != 7:
                continue
            elements = row.split(',')
            #print(elements)
            score = elements[6]
            city = elements[4]
            gender = elements[3]
            if score != '':
                scores.append(float(score) * 2)
                # if float(score) * 2 > 7:
                #     positive_text += comment
                # elif float(score) * 2 < 4:
                #     negative_text += comment
            if city != '':  # 去掉城市名为空的值
                cities.append(city)
            if gender != '':
                genders.append(gender)
    
    # 按0-10进行排序
    #print(scores)
    score_data = Counter(scores).most_common()
    score_data = sorted(score_data)
    gender_data = Counter(genders).most_common()
    print(gender_data)
    #print(score_data)
    # 定义样式
    style = Style(
        title_color='#fff',
        title_pos='center',
        width=800,
        height=600,
        background_color='#404a59'
    )


    # 根据评分数据生成柱状图
    bar = Bar('《海上钢琴师》各评分数量', '数据来源：采集自猫眼',
              title_pos='center', width=900, height=600)
    attr, value = bar.cast(score_data)
    #print(value)
    # line = Line()
    # line.add('', attr, value)
    bar.add('', attr, value, is_visualmap=True, visual_range=[0, 3500], visual_text_color='#fff', is_more_utils=True,
            is_label_show=True)
    overlap = Overlap()
    overlap.add(bar)
    # overlap.add(line)
    overlap.show_config()
    overlap.render(
        r'C:\Users\think\Desktop\情感分析\picture\评分数量-柱状图.html')

    # 对城市数据和坐标文件中的地名进行处理
    
    handle(cities)
    data = Counter(cities).most_common()  # 使用Counter类统计出现的次数，并转换为元组列表
    #print(data)
# 根据城市数据生成地理坐标图
    geo = Geo('观众地理分布', '数据来源：采集自猫眼', **style.init_style)
    attr, value = geo.cast(data)
    # print(attr)
    # print(value)
    geo.add('', attr, value, visual_range=[0, 600],maptype='china',
            visual_text_color='#fff', symbol_size=7,
            is_visualmap=True, is_piecewise=True, visual_split_number=10)
    geo.render(
         r'C:\Users\think\Desktop\情感分析\picture\观众地理分布-地理坐标图.html')

 # 根据城市数据生成柱状图
    data_top20 = Counter(cities).most_common(20)  # 返回出现次数最多的20条
    bar = Bar('观众来源排行TOP20', '数据来源：采集自猫眼',
              title_pos='center', width=1200, height=600)
    attr, value = bar.cast(data_top20)
    bar.add('', attr, value, is_visualmap=True, visual_range=[0, 3500], visual_text_color='#fff', is_more_utils=True,
            is_label_show=True)
    bar.render(r'C:\Users\think\Desktop\情感分析\picture\观众来源排行-柱状图.html')

#生成观众性别分布图
    # 设置主标题与副标题，标题设置居中，设置宽度为900
    pie = Pie("观众性别分布图", "数据来源：采集自猫眼",title_pos='center',width=900)
    attr, value = geo.cast(gender_data)
    print(value)
    attr = ["其他","男","女"]
    # 加入数据，设置坐标位置为【25，50】，上方的colums选项取消显示
    '''
    pie.add("", ["其他","男","女"], value ,visual_range=[0, 3500],
    is_legend_show=False, is_label_show=True, is_more_utils=True)
    '''
    pie.add("", attr, value, is_label_show=True, is_more_utils=True)
    # 保存图表
    
    

# 处理地名数据，解决坐标文件中找不到地名的问题
def handle(cities):
    # print(len(cities), len(set(cities)))

    # 获取坐标文件中所有地名
    data = None
    with open(
            r'D:\Anaconda3\Lib\site-packages\pyecharts\datasets\city_coordinates.json',
            mode='r', encoding='utf-8') as f:
        data = json.loads(f.read())  # 将str转换为json

    # 循环判断处理
    data_new = data.copy()  # 拷贝所有地名数据
    for city in set(cities):  # 使用set去重
        # 处理地名为空的数据
        if city == '':
            while city in cities:
                cities.remove(city)
        count = 0
        for k in data.keys():
            count += 1
            if k == city:
                break
            if k.startswith(city):  # 处理简写的地名，如 达州市 简写为 达州
                # print(k, city)
                data_new[city] = data[k]
                break
            if k.startswith(city[0:-1]) and len(city) >= 3:  # 处理行政变更的地名，如县改区 或 县改市等
                data_new[city] = data[k]
                break
        # 处理不存在的地名
        if count == len(data):
            while city in cities:
                cities.remove(city)

    # print(len(data), len(data_new))

    # 写入覆盖坐标文件
    with open(
            r'D:\Anaconda3\Lib\site-packages\pyecharts\datasets\city_coordinates.json',
            mode='w', encoding='utf-8') as f:
        f.write(json.dumps(data_new, ensure_ascii=False))  # 将json转换为str

render()

'''
[(0.0, 30), (1.0, 264), (2.0, 75), (3.0, 36), (4.0, 42), (5.0, 85), (6.0, 150), (7.0, 110), (8.0, 525), (9.0, 877), (10.0, 6581)]
[30, 264, 75, 36, 42, 85, 150, 110, 525, 877, 6581]
'''