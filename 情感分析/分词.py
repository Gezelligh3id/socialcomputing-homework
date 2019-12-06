import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys
from string import punctuation
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
# 定义要删除的标点等字符
add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc=punctuation+add_punc
 
#def cut_words(sentence):
    #print sentence
#    return " ".join(jieba.cut(sentence)).encode('utf-8')
# 指定要分词的文本
f=codecs.open(r'C:\Users\think\Desktop\情感分析\demo.txt','r',encoding="utf8")
#指定分词结果的保存文本
target = codecs.open(r'C:\Users\think\Desktop\情感分析\answer.txt', 'w',encoding="utf8")
print ('open files')
line_num=1
line = f.readline()
while line:
    print('---- processing ', line_num, ' article----------------')
    # 第一次分词，用于移除标点等符号
	#line=re.sub(r'[A-Za-z0-9]|/d+','',line)   
    # #用于移除英文和数字
    line_seg = " ".join(jieba.cut(line))
    # 移除标点等需要删除的符号
    testline=line_seg.split(' ')
    te2=[]
    for i in testline:
        te2.append(i)
        if i in all_punc:
            te2.remove(i)
    # 返回的te2是个list，转换为string后少了空格，因此需要再次分词
	# 第二次在仅汉字的基础上再次进行分词
    line_seg2 = " ".join(jieba.cut(''.join(te2)))
    target.writelines(line_seg2)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()
exit()