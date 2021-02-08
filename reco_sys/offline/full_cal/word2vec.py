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

from offline import SparkSessionBase
# 导入其他人工依赖函数，需要注意default文件不能有执行性语句
from setting.default import channelInfo, segmentation
from pyspark.ml.feature import Word2Vec


class TrainWord2VecModel(SparkSessionBase):

    SPARK_APP_NAME = "Word2Vec"
    SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


w2v = TrainWord2VecModel()


# 这里训练一个频道模型演示即可
w2v.spark.sql("use di_news_db_test")
article = w2v.spark.sql("select * from article_data where channel_id=17 limit 3")
words_df = article.rdd.mapPartitions(segmentation).toDF(['article_id', 'channel_id', 'words'])  # 分词
words_df.show()

# 训练模型与保存
"""
vectorSize：词向量维度
minCount:最少词汇频数，频数少于该值的词汇不参与词向量训练
numPartitions参数,这个参数默认是1,如果使用默认参数,等于只有一个job进行fit,如果数据很大,这个过程将会非常漫长
"""
new_word2Vec = Word2Vec(vectorSize=100, inputCol="words", outputCol="model", minCount=3)
new_model = new_word2Vec.fit(words_df)
new_model.write().overwrite().save("hdfs://nameserviceHA/data/tmp/models/test_C17.word2vec")


# 加载某个频道模型，得到每个词的向量
from pyspark.ml.feature import Word2VecModel
channel_id = 17
channel = "前端"
wv_model = Word2VecModel.load("hdfs://nameserviceHA/data/tmp/models/channel_%d_%s.word2vec" % (channel_id, channel))
vectors = wv_model.getVectors()  # 单个元素即”词汇+词向量“的形式

# 获取新增的文章画像，得到文章画像的关键词、主题词
# 选出新增的文章的画像做测试，上节计算的画像中有不同频道的，我们选取Python频道的进行计算测试
"""
article_profile：
+----------+----------+--------------------------------------------+
|article_id|channel_id|                         keywords|    topics|
+----------+----------+--------------------+-----------------------+
|    141462|         3|            {美丽:5.46, 心情:4.87}| [心情,美丽]|
+----------+----------+-------------------------------------------+
"""
articleProfile = w2v.spark.sql("select * from article_profile where channel_id=17 limit 10")
profile = articleProfile.filter('channel_id = {}'.format(channel_id))
# 将articleProfile的keywords属性字典拆分，与vectors进行内连接，获取该channel所有词汇的词向量
profile.registerTempTable("incremental")
articleKeywordsWeights = w2v.spark.sql("select article_id, channel_id, keyword, weight from incremental LATERAL VIEW explode(keywords) AS keyword, weight")
# _article_profile包含(article_id, channel_id, keyword, weight, vector)
_article_profile = articleKeywordsWeights.join(vectors, vectors.word == articleKeywordsWeights.keyword, "inner")

# 这里用词的权重 * 词的向量 = weights x vector=new_vector
articleKeywordVectors = _article_profile.rdd.map(lambda row: (row.article_id, row.channel_id, row.keyword, row.weight * row.vector)).toDF(["article_id", "channel_id", "keyword", "weightingVector"])
articleKeywordVectors.show()


# 计算得到文章的平均词向量即文章的向量
def avg(row):
    x = 0
    for v in row.vectors:
        x += v
    #  将平均向量作为article的向量
    return row.article_id, row.channel_id, x / len(row.vectors)


articleKeywordVectors.registerTempTable("tempTable")
articleVector = w2v.spark.sql(
    "select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors from tempTable group by article_id").rdd.map(
    avg).toDF(["article_id", "channel_id", "articleVector"])


# 对计算出的”articleVector“列进行处理，该列为Vector类型，不能直接存入HIVE，HIVE不支持该数据类型
def toArray(row):
    return row.article_id, row.channel_id, [float(i) for i in row.articleVector.toArray()]


articleVector = articleVector.rdd.map(toArray).toDF(['article_id', 'channel_id', 'articleVector'])
articleVector.show()

# 文章向量保存于article_vector
articleVector.write.insertInto("article_vector")


# 计算文章相似度minihashing
from pyspark.ml.linalg import Vectors
# 选取部分数据做测试，article_vector(article_id, articlevector)
w2v.spark.sql("use di_news_db_test")
article_vector = w2v.spark.sql("select article_id, articleVector from article_vector where channel_id=18 and article_id in (12888, 13071, 13077, 14673)")
train = article_vector.select(['article_id', 'articleVector'])


def _array_to_vector(row):
    # 数组转化为向量形式（dense方法转化为稠密向量(普通向量)）
    return row.article_id, Vectors.dense(row.articleVector)


train = train.rdd.map(_array_to_vector).toDF(['article_id', 'articleVector'])
train.show()

from pyspark.ml.feature import BucketedRandomProjectionLSH

# LSH局部敏感哈希，是一种针对海量高维数据的快速最近邻查找算法；计算文章之间的相似度
"""
InputCol 	DF中待变换的特征 	特征类型必须为：vector
OutputCol 	变换后的特征名称 	转换后的类型为：array[vector]
BucketLength 	每个哈希桶的长度 	更大的桶降可降低假阴性率
NumHashTables 	哈希表的数量 	散列表数量的增加降低了错误的否定率，并且降低它提高了运行性能，默认：1
"""
brp = BucketedRandomProjectionLSH(inputCol='articleVector', outputCol='hashes', numHashTables=4.0, bucketLength=10.0)
model = brp.fit(train)

# 5.0表示筛选距离，距离越大则文章相似度越小
similar = model.approxSimilarityJoin(train, train, 5.0, distCol='EuclideanDistance')
# 对相似文章排序
similar.sort(['EuclideanDistance']).show()


# 将计算结果保存在mysql
def save_hbase(partition):

    from setting.mysql_conn import mysql_tool, mysql_host, mysql_user, mysql_password, mysql_db, mysql_port
    my_model = mysql_tool(mysql_host, mysql_user, mysql_password, mysql_db, int(mysql_port))

    data = []
    for row in partition:
        if row.datasetA[0] == row.datasetB[0]:
            pass
        else:
            data.append((row.datasetA[0], row.datasetB[0], row.EuclideanDistance))
    print(data)
    table = 'news_article_similar'
    sql = "INSERT ignore INTO " + table + " (article_id, similar, EuclideanDistance)" + " VALUES ('%s', '%s', '%s') "
    my_model.batch_insert_tool(sql, data)


# map与foreach的区别：前者返回仍是rdd类型，后者不是
similar.rdd.foreachPartition(save_hbase)