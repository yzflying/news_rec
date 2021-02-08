#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
# BASE_DIR为offline上一级路径，避免出现后面导包问题
# 添加工程路径reco_sys平级目录,后续导入包从该目录的子目录开始导入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR))

# 指定python环境，当存在多个版本时，不指定很可能会导致出错
PYSPARK_PYTHON = "/data/anaconda3/envs/python36/bin/python"
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON

# offline文件夹下含__init__文件，此时offline类似于模块
from offline import SparkSessionBase
import pyspark


class KeywordsToTfidf(SparkSessionBase):

    SPARK_APP_NAME = "keywordsByTFIDF"
    SPARK_EXECUTOR_MEMORY = "7g"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


# 分词功能实现
def segmentation(partition):
    import os
    import re

    import jieba
    import jieba.analyse
    import jieba.posseg as pseg   # pseg.cut不能写成jieba.posseg.cut形式；因jieba下没有__init__.py文件
    import codecs  # 编码转换器模块

    # 文件路径，放集群上，集群(hdfs://nameserviceHA/)路径
    # load_userdict函数无法访问集群文件，此处选择放本地
    abspath = "/data/yzx/data/words"

    # 结巴加载用户词典
    userDict_path = os.path.join(abspath, "ITKeywords.txt")
    jieba.load_userdict(userDict_path)

    # 停用词文本
    stopwords_path = os.path.join(abspath, "stopwords.txt")

    def get_stopwords_list():
        """返回stopwords列表,编码问题open文件时加上 encoding='utf-8'"""
        stopwords_list = [i.strip() for i in codecs.open(stopwords_path, encoding='utf-8').readlines()]
        return stopwords_list

    # 所有的停用词列表
    stopwords_list = get_stopwords_list()

    # 分词
    def cut_sentence(sentence):
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        # print(sentence,"*"*100)
        # 带词性的分词 eg:[(i.word, i.flag), pair('今天', 't'), pair('有', 'd'), pair('雾', 'n'), pair('霾', 'g')]
        seg_list = pseg.lcut(sentence)
        # 去除停用词
        seg_list = [i for i in seg_list if i.word not in stopwords_list]
        filtered_words_list = []
        for seg in seg_list:
            # 去除长度小于2的词
            if len(seg.word) <= 1:
                continue
            # 如果是英文名词，保留长度大于2的单词
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            # 如果是中文名词，保存
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            # 是自定一个词语或者是英文单词，保存
            elif seg.flag in ["x", "eng"]:
                filtered_words_list.append(seg.word)
        # 返回词汇列表
        return filtered_words_list

    for row in partition:
        sentence = re.sub("<.*?>", "", row.sentence)    # 将article_data的sentence列替换掉标签数据
        words = cut_sentence(sentence)
        yield row.article_id, row.channel_id, words


ktt = KeywordsToTfidf()
"""
将article_data数据利用函数segmentation分词，得到words_df

article.article_data(含"article_id", "channel_id", "channel_name", "title", "article_content","sentence")
141462 3 ios test-20190316-115123 今天天气不错，心情很美丽！！！ ios,test-20190316-115123,今天天气不错，心情很美丽！！！

words_df(含"article_id", "channel_id", "words")
+----------+----------+--------------------------+
|article_id|channel_id|                     words|
+----------+----------+--------------------------+
|        11|         1|            [liked, movie]|
"""
# 读取 article_data 文章原始数据，得到article_dataframe，分词后得到words_df，含"article_id", "channel_id", "words"三个属性
ktt.spark.sql("use di_news_db_test")
article_dataframe = ktt.spark.sql("select * from article_data limit 5")
words_df = article_dataframe.rdd.mapPartitions(segmentation).toDF(["article_id", "channel_id", "words"])
words_df.show()


# 词语与词频统计
from pyspark.ml.feature import CountVectorizer
# 总词汇的大小，文本中必须出现的次数；输入属性列words
cv = CountVectorizer(inputCol="words", outputCol="countFeatures", vocabSize=2000, minDF=1.0)
# 训练词频统计模型
cv_model = cv.fit(words_df)
cv_model.write().overwrite().save("hdfs://nameserviceHA/data/tmp/models/CV.model")

# 词语与词频统计
from pyspark.ml.feature import CountVectorizerModel
cv_model = CountVectorizerModel.load("hdfs://nameserviceHA/data/tmp/models/CV.model")
# 得出词频向量结果
cv_result = cv_model.transform(words_df)
cv_result.show()
"""
cv_result类似如下，在words_df的基础上增加了countFeatures列
“7”表示词汇表(按所有词汇频数从大到小排序)共含有7个单词，
[0,1]表示两个该样本两个词汇在词汇表的索引，[0,5,6]同
[1.0,1.0]表示该样本中词汇频数为1
+----------+----------+--------------------------+--------------------+
|article_id|channel_id|                     words|       countFeatures|
+----------+----------+--------------------------+--------------------+
|        11|         1|      [liked, movie]      | (7,[0,1],[1.0,1.0])|
|        22|         2|[recommend, movie, friends|(7,[0,5,6],[1.0,1...|
|        33|         3|[movie, acting, horrible] |(7,[0,2,4],[1.0,1...|
|        44|         4|[watching, movie,liked]   |(7,[0,1,3],[1.0,1...|
+----------+----------+--------------------------+--------------------+
"""

# 训练IDF模型(inputCol选择countFeatures列)
from pyspark.ml.feature import IDF
idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
idfModel = idf.fit(cv_result)
idfModel.write().overwrite().save("hdfs://nameserviceHA/data/tmp/models/IDF.model")
tfidf_result = idfModel.transform(cv_result)

# 打印词汇总表cv_model（已按频数排序）的前10个词，及其词汇对应的idf值
print(cv_model.vocabulary[:10])
print(idfModel.idf.toArray()[:10])
"""
tfidf_result的结果如下，在cv_result的基础上增加了idfFeatures列
[0.0,0.5]表示该样本两个词汇的idf值，越大代表越重要
+----------+----------+--------------------+--------------------+--------------------+
|article_id|channel_id|               words|       countFeatures|         idfFeatures|
+----------+----------+--------------------+--------------------+--------------------+
|        11|         1|      [liked, movie]| (7,[0,1],[1.0,1.0])|(7,[0,1],[0.0,0.5...|
|        22|         2|[recommend, movie...|(7,[0,2,3],[1.0,1...|(7,[0,2,3],[0.0,0...|
|        33|         3|[movie, acting, h...|(7,[0,4,5],[1.0,1...|(7,[0,4,5],[0.0,0...|
|        44|         4|[watching, movie,...|(7,[0,1,6],[1.0,1...|(7,[0,1,6],[0.0,0...|
+----------+----------+--------------------+--------------------+--------------------+
"""

# 持久化保存词汇、idf值、索引，三者对应关系，并保存于idf_keywords_values
keywords_list_with_idf = list(zip(cv_model.vocabulary, idfModel.idf.toArray()))
"""
keywords_list_with_idf 形如：
[('movie', 0.0), ('liked', 0.5108256237659907), ('recommend', 0.9162907318741551), 
('friends', 0.9162907318741551), ('acting', 0.9162907318741551), 
('horrible', 0.9162907318741551), ('watching', 0.9162907318741551)]
"""


def func(data):
    for index in range(len(data)):
        data[index] = list(data[index])
        data[index].append(index)
        data[index][1] = float(data[index][1])


"""
将 keywords_list_with_idf 的每个元组形式的元素(word, idf_value)转化为列表形式[word, idf_value, index]
[('movie', 0.0, 0), ('liked', 0.5108256237659907, 1), ('recommend', 0.9162907318741551, 2), 
('friends', 0.9162907318741551, 3), ('acting', 0.9162907318741551, 4), 
('horrible', 0.9162907318741551, 5), ('watching', 0.9162907318741551, 6)]
"""
func(keywords_list_with_idf)


sc = ktt.spark.sparkContext
rdd = sc.parallelize(keywords_list_with_idf)   # parallelize 并行，将普通数组转化成RDD格式，方便后续并行计算
df = rdd.toDF(["keywords", "idf", "index"])

df.write.insertInto('idf_keywords_values')


# 计算tfidf值
# from pyspark.ml.feature import CountVectorizerModel
# cv_model = CountVectorizerModel.load("hdfs://hadoop-master:9000/headlines/models/countVectorizerOfArticleWords.model")
# from pyspark.ml.feature import IDFModel
# idf_model = IDFModel.load("hdfs://hadoop-master:9000/headlines/models/IDFOfArticleWords.model")
# cv_result = cv_model.transform(words_df)
# tfidf_result = idf_model.transform(cv_result)


def func(partition):
    TOPK = 10
    for row in partition:
        # 找到索引与IDF值并进行排序
        _ = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _ = sorted(_, key=lambda x: x[1], reverse=True)
        result = _[:TOPK]
        for word_index, tfidf in result:
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)


"""
tfidf_result(含属性 article_id,channel_id,words,countFeatures,idfFeatures)中的idfFeatures，按照索引对应的idf值排序，每篇文章取前10个词，得到_keywordsByTFIDF
tfidf_result：
+----------+----------+--------------------+--------------------+--------------------+
|article_id|channel_id|               words|       countFeatures|         idfFeatures|
+----------+----------+--------------------+--------------------+--------------------+
|        11|         1|      [liked, movie]| (7,[0,1],[1.0,1.0])|(7,[0,1],[0.0,0.5...|
|        22|         2|[recommend, movie...|(7,[0,2,3],[1.0,1...|(7,[0,2,3],[0.0,0...|
|        33|         3|[movie, acting, h...|(7,[0,4,5],[1.0,1...|(7,[0,4,5],[0.0,0...|
|        44|         4|[watching, movie,...|(7,[0,1,6],[1.0,1...|(7,[0,1,6],[0.0,0...|
+----------+----------+--------------------+--------------------+--------------------+
_keywordsByTFIDF：
+----------+----------+--------------------+--------------------+
|article_id|channel_id|               index|               tfidf|
+----------+----------+--------------------+--------------------+
|        11|         1|                   1|              0.5108|
|        11|         1|                   0|                 0.0|
|        22|         2|                   2|              0.9162|
|        22|         2|                   3|              0.9162|
|        22|         2|                   0|                 0.0|
+----------+----------+--------------------+--------------------+
"""
_keywordsByTFIDF = tfidf_result.rdd.mapPartitions(func).toDF(["article_id", "channel_id", "index", "tfidf"])


"""
idf_keywords_values:
+--------------------+--------------------+----------------+
|             keyword|               tfidf|           index|
+--------------------+--------------------+----------------+
|               liked|              0.5108|               1|
|               movie|                 0.0|               0|
|           recommend|              0.9162|               2|
|             friends|              0.9162|               3|
+--------------------+--------------------+----------------+
keywordsIndex:
+--------------------+----------------+
|             keyword|           index|
+--------------------+----------------+
|               liked|               1|
|               movie|               0|
|           recommend|               2|
|             friends|               3|
+--------------------+----------------+
tfidf_keywords_values：将_keywordsByTFIDF中的词汇索引换成实际的词汇:
+----------+----------+--------------------+--------------------+
|article_id|channel_id|             keyword|               tfidf|
+----------+----------+--------------------+--------------------+
|        11|         1|               liked|              0.5108|
|        11|         1|               movie|                 0.0|
|        22|         2|           recommend|              0.9162|
|        22|         2|             friends|              0.9162|
|        22|         2|               movie|                 0.0|
+----------+----------+--------------------+--------------------+
"""
# 利用结果索引与”idf_keywords_values“合并知道词
keywordsIndex = ktt.spark.sql("select keyword, index idx from idf_keywords_values")
# 利用结果索引与”idf_keywords_values“合并知道词
keywordsByTFIDF = _keywordsByTFIDF.join(keywordsIndex, keywordsIndex.idx == _keywordsByTFIDF.index).select(["article_id", "channel_id", "keyword", "tfidf"])
keywordsByTFIDF.write.insertInto("tfidf_keywords_values")
