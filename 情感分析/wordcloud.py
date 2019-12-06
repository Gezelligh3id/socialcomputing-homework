# #coding=utf-8
# import matplotlib.pyplot as plt
# from scipy.misc import imread
#from wordcloud import WordCloud
import jieba, codecs
# from collections import Counter
# import numpy as np
# from PIL import Image # 图像处理库
# import wordcloud # 词云展示库

# text = codecs.open(r'C:\Users\think\Desktop\情感分析\answer.txt', 'r', encoding='utf-8').read()
# text_jieba = list(jieba.cut(text))
# c = Counter(text_jieba)  # 计数
# word = c.most_common(800)  # 取前500
# print(word)
# bg_pic = imread(r'C:\Users\think\Desktop\情感分析\beijing.jpg')
# wc = WordCloud(
#     #font_path='C:\Windows\Fonts\SIMYOU.TTF',  # 指定中文字体
#     background_color='white',  # 设置背景颜色
#     max_words=2000,  # 设置最大显示的字数
#     mask=bg_pic,  # 设置背景图片
#     max_font_size=200,  # 设置字体最大值
#     random_state=20  # 设置多少种随机状态，即多少种配色
# )
# wc.generate_from_frequencies(dict(word))  # 生成词云
 
 
# wc.to_file('result.jpg')
 
# # show
# plt.imshow(wc)
# plt.axis("off")
# plt.figure()
# plt.imshow(bg_pic, cmap=plt.cm.gray)
# plt.axis("off")
# plt.show()

# mask = np.array(Image.open(r'C:\Users\think\Desktop\情感分析\beijing.jpg')) # 定义词频背景
# wc = wordcloud.WordCloud(
#     font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
#     mask=mask, # 设置背景图
#     max_words=200, # 最多显示词数
#     max_font_size=100 # 字体最大值
# )

# wc.generate_from_frequencies(word) # 从字典生成词云
# image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
# wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
# plt.imshow(wc) # 显示词云
# plt.axis('off') # 关闭坐标轴
# plt.show() # 显示图像

# 导入扩展库
import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库

# 读取文件
text = codecs.open(r'C:\Users\think\Desktop\情感分析\doc\answer.txt', 'r', encoding='utf-8').read()
print(text)
#fn = open(r'C:\Users\think\Desktop\情感分析\answer.txt') # 打开文件
#string_data = fn.read() # 读出整个文件
#fn.close() # 关闭文件

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"') # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', text) # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(text, cut_all = False) # 精确模式分词
object_list = []
remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                u'通常',u'如果',u'我们',u'需要',u'他',u'我',u'看',u'电影',u'很',u'人',u'不',u'有',u'也',u'这',u'就',u'…',u'\r\n',u'还是',u'就',u'就是'] # 自定义去除词库

for word in seg_list_exact: # 循环读出每个分词
    if word not in remove_words: # 如果不在去除词库中
        object_list.append(word) # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list) # 对分词做词频统计
print(word_counts)
word_counts_top10 = word_counts.most_common(10) # 获取前10最高频的词
print (word_counts_top10) # 输出检查

# 词频展示
#mask = np.array(Image.open(r'C:\Users\think\Desktop\情感分析\blackbeijing.jpg')) # 定义词频背景
wc = wordcloud(
    font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
    width=2000,height=1200,
    background_color='white',  # 设置背景颜色
    #mask=mask, # 设置背景图
    max_words=200, # 最多显示词数
    max_font_size=300, # 字体最大值
    random_state=30  # 设置多少种随机状态，即多少种配色
)
# wc = WordCloud(
#     #font_path='C:\Windows\Fonts\SIMYOU.TTF',  # 指定中文字体
#     background_color='white',  # 设置背景颜色
#     max_words=2000,  # 设置最大显示的字数
#     mask=bg_pic,  # 设置背景图片
#     max_font_size=200,  # 设置字体最大值
#     random_state=20  # 设置多少种随机状态，即多少种配色
# )
# wc.generate_from_frequencies(dict(word))  # 生成词云
wc.generate_from_frequencies(word_counts) # 从字典生成词云
#image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
#wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
plt.imshow(wc) # 显示词云
plt.axis('off') # 关闭坐标轴
plt.show() # 显示图像
wc.to_file(r'C:\Users\think\Desktop\情感分析\result.jpg')