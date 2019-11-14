# -*- coding: utf-8 -*-
# @Time : 2019/11/13 14:55
# @FileName: word2vec-gensim.py
# @Software: PyCharm
# @Author : yip
# @Email : 522364642@qq.com
# @Blog : https://blog.csdn.net/qq_30189255
# @Github : https://github.com/yip522364642


import warnings

warnings.filterwarnings("ignore")

'''
1 获取文本语料并查看
'''
# with open('text8', 'r', encoding='utf-8') as file:
#     for line in file.readlines():
#         print(line)

'''
2 载入数据，训练并保存模型
'''
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 输出日志信息
sentences = word2vec.Text8Corpus('text8')  # 将语料保存在sentence中
model = word2vec.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)  # 生成词向量空间模型
model.save('text8_word2vec.model')  # 保存模型


'''
3 加载模型，实现各个功能
'''
# 加载模型
model = word2vec.Word2Vec.load('text8_word2vec.model')

# 计算两个词的相似度/相关程度
print("计算两个词的相似度/相关程度")
word1 = 'man'
word2 = 'woman'
result1 = model.similarity(word1, word2)
print(word1 + "和" + word2 + "的相似度为：", result1)
print("\n================================")

# 计算某个词的相关词列表
print("计算某个词的相关词列表")
word = 'bad'
result2 = model.most_similar(word, topn=10)  # 10个最相关的
print("和" + word + "最相关的词有：")
for item in result2:
    print(item[0], item[1])
print("\n================================")

# 寻找对应关系
print("寻找对应关系")
print(' "boy" is to "father" as "girl" is to ...? ')
result3 = model.most_similar(['girl', 'father'], ['boy'], topn=3)
for item in result3:
    print(item[0], item[1])
print("\n")

more_examples = ["she her he", "small smaller bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print("\n================================")

# 寻找不合群的词
print("寻找不合群的词")
result4 = model.doesnt_match("flower grass pig tree".split())
print("不合群的词：", result4)
print("\n================================")

# 查看词向量（只在model中保留中的词）
print("查看词向量（只在model中保留中的词）")
word = 'girl'
print(word, model[word])
# for word in model.wv.vocab.keys():  # 查看所有单词
#     print(word, model[word])


'''
4 增量训练
'''
model = word2vec.Word2Vec.load('text8_word2vec.model')
more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)
model.save('text8_word2vec.model')
