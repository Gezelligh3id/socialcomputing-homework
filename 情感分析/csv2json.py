#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import csv
#import sys
#df = pd.read_csv(r'C:\Users\think\Desktop\情感分析\doc\maoyan1.csv')
#print(df)
#print(df.sum(columns = ['score']))
#label = pd.read_csv(r'D:\test.csv')
#print(df.axis,inplace=True)
#df.drop(df.index[1],axis=1,inplace= True)
#print(df)
#print(label)
#label3 = label[label.class_id.isin([3])]
import sys
import json

input_file = r'C:\Users\think\Desktop\情感分析\doc\maoyan.csv'
lines = ""
# 读取文件
with open(input_file, "r",encoding='utf-8') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
keys = lines[0].split(',') 
line_num = 1
total_lines = len(lines)
# 数据存储
datas = []
while line_num < total_lines:
        values = lines[line_num].split(",")
        datas.append(dict(zip(keys, values)))
        line_num = line_num + 1
# 序列化时对中文默认使用的ascii编码.想输出真正的中文需要指定ensure_ascii=False
json_str = json.dumps(datas, ensure_ascii=False, indent=4)
# 去除\",\\N,\n 无关符号
result_data = json_str.replace(r'\"','').replace(r'\\N','').replace(r'\n','')
output_file = input_file.replace("csv", "json")
# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result_data)
    print("convert success")