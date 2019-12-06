# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:37:48 2019

@author: think
"""

import numpy as np
#import snownlp
import os
import pandas as pd
import codecs
from snownlp import SnowNLP
import matplotlib.pyplot as plt
text = codecs.open(r'C:\Users\think\Desktop\情感分析\doc\demo.txt', 'r', encoding='utf-8').read()
#zinput = str(input("Enter the word you want me to search: "))
list = []
with open(r'C:\Users\think\Desktop\情感分析\doc\answer.txt', 'r', encoding='utf-8') as file:
    for line in file:    
        list.append(line)
#print(list)
#f = open(r'C:\Users\think\Desktop\情感分析\doc\answer.txt', 'r')
#list = f.readlines()

def file_do(list_info, flag):
        # 获取文件大小
        # file = 'C:\Users\think\Desktop\情感分析\doc\a\\'[:-1] + list_info + '.csv'
        if flag == 0:
            file_size = os.path.getsize(r'C:\Users\think\Desktop\情感分析\doc\all.csv')
            if file_size == 0:
            # 表头
                name = ['content','score']
            # 建立DataFrame对象
                file_test = pd.DataFrame(columns=name, data=list_info)
            # 数据写入
                file_test.to_csv(r'C:\Users\think\Desktop\情感分析\doc\all.csv',
                             encoding='utf_8_sig', index=False)
            else:
                with open(r'C:\Users\think\Desktop\情感分析\doc\all.csv', 'a+',
                             encoding='utf_8_sig') as file_test:
                # 追加到文件后面
                    writer = csv.writer(file_test)
                # 写入文件
                    writer.writerows(list_info)
                    
                    
        if flag == 1:
            file_size = os.path.getsize(r'C:\Users\think\Desktop\情感分析\doc\neg.csv')
            if file_size == 0:
            # 表头
                name = ['content','score']
            # 建立DataFrame对象
                file_test = pd.DataFrame(columns=name, data=list_info)
            # 数据写入
                file_test.to_csv(r'C:\Users\think\Desktop\情感分析\doc\neg.csv',
                             encoding='utf_8_sig', index=False)
            else:
                with open(r'C:\Users\think\Desktop\情感分析\doc\neg.csv', 'a+',
                             encoding='utf_8_sig') as file_test:
                # 追加到文件后面
                    writer = csv.writer(file_test)
                # 写入文件
                    writer.writerows(list_info)

        if flag == 2:
            file_size = os.path.getsize(r'C:\Users\think\Desktop\情感分析\doc\pos.csv')
            if file_size == 0:
            # 表头
                name = ['content','score']
            # 建立DataFrame对象
                file_test = pd.DataFrame(columns=name, data=list_info)
            # 数据写入
                file_test.to_csv(r'C:\Users\think\Desktop\情感分析\doc\pos.csv',
                             encoding='utf_8_sig', index=False)
            else:
                with open(r'C:\Users\think\Desktop\情感分析\doc\pos.csv', 'a+',
                             encoding='utf_8_sig') as file_test:
                # 追加到文件后面
                    writer = csv.writer(file_test)
                # 写入文件
                    writer.writerows(list_info)
sentimentslist = []
list_all = []
list_neg = []
list_pos = []
count = 0
for i in list:
    if not i is '\n':
        s = SnowNLP(i)
        #print(i)
        #print(s.sentiments)
        print(count)
        count = count + 1
        #if count == 10:
        #    break
        sentimentslist.append(s.sentiments)
        list_tmp = [i, s.sentiments]
        list_all.append(list_tmp)
        if float(s.sentiments) < 0.5:
            list_neg.append(list_tmp)
        else:
            list_pos.append(list_tmp)
file_do(list_all, 0)
file_do(list_neg, 1)
file_do(list_pos, 2)
plt.hist(sentimentslist, bins = np.arange(0, 1, 0.01), facecolor = 'g')
plt.xlabel('Sentiments Probability')
plt.ylabel('Quantity')
plt.title('使用SnowNLP对《海上钢琴师》进行情感分析')
plt.show()
