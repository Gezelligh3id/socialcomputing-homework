# #coding:UTF-8
# import sys
# from snownlp import SnowNLP
# import jieba
# import jieba.analyse
# import jieba.posseg as pseg
# import codecs,sys
# from string import punctuation
# def read_and_analysis():
#   #f = open(r'C:\Users\think\Desktop\demo1.txt',"r",encoding="utf8")
#   f = codecs.open(r'C:\Users\think\Desktop\demo1.txt','r',encoding="utf8")
#   #print(1)
#   #fw = open(r'C:\Users\think\Desktop\result.txt', "w",encoding="utf8")
#   fw = codecs.open(r'C:\Users\think\Desktop\answer.txt', 'w',encoding="utf8")
#   #print(2)
#   line_num = 1
#   line = f.readline()
#   while line:
#     print('---- processing ', line_num, ' article----------------')
#     lines = line.strip().split("\t")
#     #if len(lines) < 2:
#     #  continue

#     s = SnowNLP(lines[1].decode('utf-8'))
#     # s.words 查询分词结果
#     seg_words = ""
#     for x in s.words:
#       seg_words += "_"
#       seg_words += x
#     # s.sentiments 查询最终的情感分析的得分
#     print(s.sentiments)
#     fw.write(lines[0] + "\t" + lines[1] + "\t" + seg_words.encode('utf-8') + "\t" + str(s.sentiments) + "\n")
#     line_num = line_num + 1
#     line = f.readline()
#   fw.close()
#   f.close()

# if __name__ == "__main__":
#   #input_file = sys.argv[1]
#   #output_file = sys.argv[2]
#   read_and_analysis()

# 导入SnowNLP库
from snownlp import SnowNLP

# 需要操作的句子
text = 'emm挺好看的，但是我要是他兄弟我会把他打晕捞出来。看完这个电影觉得他不下陆地就是懦弱。'

s = SnowNLP(text)

# 分词
print(s.words)

print(s.sentiments)