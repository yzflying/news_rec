#! /usr/bin/env python
# -*- coding: utf-8 -*-


# 文章离线画像增量更新

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
from datetime import datetime
from datetime import timedelta
import pyspark.sql.functions as F
import gc
# from offline.full_cal.compute_tfidf import segmentation


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


class UpdateArticle(SparkSessionBase):
    """
    更新文章画像
    """
    SPARK_APP_NAME = "updateArticle"
    ENABLE_HIVE_SUPPORT = True

    SPARK_EXECUTOR_MEMORY = "7g"

    def __init__(self):
        self.spark = self._create_spark_session()
        self.cv_path = "hdfs://nameserviceHA/data/tmp/models/countVectorizerOfArticleWords.model"
        self.idf_path = "hdfs://nameserviceHA/data/tmp/models/IDFOfArticleWords.model"

    def get_cv_model(self):
        # 词语与词频统计
        from pyspark.ml.feature import CountVectorizerModel
        cv_model = CountVectorizerModel.load(self.cv_path)
        return cv_model

    def get_idf_model(self):
        from pyspark.ml.feature import IDFModel
        idf_model = IDFModel.load(self.idf_path)
        return idf_model

    @staticmethod
    def compute_keywords_tfidf_topk(words_df, cv_model, idf_model):
        # 定义静态方法，无需实例化对象UpdateArticle即可直接调用
        """保存tfidf值高的20个关键词
        :param words_df:
        +----------+----------+---------------------------+
        |article_id|channel_id|                      words|
        +----------+----------+---------------------------+
        |        11|         1|             [liked, movie]|
        |        22|         2|[recommend, movie, friends]|
        :return:_keywordsByTFIDF：
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
        cv_result = cv_model.transform(words_df)
        tfidf_result = idf_model.transform(cv_result)

        # print("transform compelete")

        # 取TOP-N的TFIDF值高的结果
        def func(partition):
            TOPK = 20
            for row in partition:
                _ = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
                _ = sorted(_, key=lambda x: x[1], reverse=True)
                result = _[:TOPK]
                #         words_index = [int(i[0]) for i in result]
                #         yield row.article_id, row.channel_id, words_index

                for word_index, tfidf in result:
                    yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)

        _keywordsByTFIDF = tfidf_result.rdd.mapPartitions(func).toDF(["article_id", "channel_id", "index", "tfidf"])

        return _keywordsByTFIDF

    def merge_article_data(self):
        """
        合并业务中增量更新的文章数据，功能类似于 merge_data.py 文件实现
        :return: 增量更新的文章，(含"article_id", "channel_id", "channel_name", "title", "article_content","sentence")
        """
        # 获取文章相关数据, 指定过去一个小时整点到整点的更新数据
        # 如：26日：1：00~2：00，2：00~3：00，左闭右开
        self.spark.sql("use di_news_db_test")
        # _yester = datetime.today().replace(minute=0, second=0, microsecond=0)
        _yester = datetime.today().replace(year=2019, month=3, day=20, hour=17, minute=0, second=0, microsecond=0)
        start = datetime.strftime(_yester + timedelta(days=0, hours=-1, minutes=0), "%Y-%m-%d %H:%M:%S")
        end = datetime.strftime(_yester, "%Y-%m-%d %H:%M:%S")

        # 合并后保留：article_id、channel_id、channel_name、title、content
        # +----------+----------+--------------------+--------------------+
        # | article_id | channel_id | title | content |
        # +----------+----------+--------------------+--------------------+
        # | 141462 | 3 | test - 20190316 - 115123 | 今天天气不错，心情很美丽！！！ |
        basic_content = self.spark.sql(
            "select a.article_id, a.channel_id, a.title, b.content from news_article_basic a "
            "inner join news_article_content b on a.article_id=b.article_id where a.update_time >= '{}' "
            "and a.update_time < '{}' and a.status = 2".format(start, end))

        # 增加channel的名字，后面会使用
        basic_content.registerTempTable("temparticle")
        channel_basic_content = self.spark.sql(
            "select t.*, n.channel_name from temparticle t left join news_channel n on t.channel_id=n.channel_id")

        # 利用concat_ws方法，将多列数据合并为一个长文本内容（频道，标题以及内容合并）
        sentence_df = channel_basic_content.select("article_id", "channel_id", "channel_name", "title", "content", \
                                                   F.concat_ws(
                                                       ",",
                                                       channel_basic_content.channel_name,
                                                       channel_basic_content.title,
                                                       channel_basic_content.content
                                                   ).alias("sentence")
                                                   )
        del basic_content
        del channel_basic_content
        gc.collect()

        sentence_df.write.insertInto("article_data")
        return sentence_df

    def generate_article_label(self, sentence_df):
        """
        生成文章标签  tfidf, textrank
        :param sentence_df: 增量更新的文章 article_data，(含"article_id", "channel_id", "channel_name", "title", "article_content","sentence")
        :return:
        """
        # 进行分词
        words_df = sentence_df.rdd.mapPartitions(segmentation).toDF(["article_id", "channel_id", "words"])
        cv_model = self.get_cv_model()
        idf_model = self.get_idf_model()

        # 1、保存所有的词的idf的值，利用idf中的词的标签索引
        # 工具与业务隔离
        _keywordsByTFIDF = UpdateArticle.compute_keywords_tfidf_topk(words_df, cv_model, idf_model)
        print("***********")
        print(_keywordsByTFIDF)
        """
        keywordsIndex:
        +--------------------+----------------+
        |             keyword|             idx|
        +--------------------+----------------+
        |               liked|               1|
        |               movie|               0|
        |           recommend|               2|
        |             friends|               3|
        +--------------------+----------------+
        keywordsByTFIDF:
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
        keywordsIndex = self.spark.sql("select keyword, index idx from idf_keywords_values")

        keywordsByTFIDF = _keywordsByTFIDF.join(keywordsIndex, keywordsIndex.idx == _keywordsByTFIDF.index).select(
            ["article_id", "channel_id", "keyword", "tfidf"])

        keywordsByTFIDF.write.insertInto("tfidf_keywords_values")

        del cv_model
        del idf_model
        del words_df
        del _keywordsByTFIDF
        gc.collect()

        # # 计算textrank
        # """
        # textrank_keywords_values:
        # +----------+----------+--------------------+--------------------+
        # |article_id|channel_id|             keyword|            textrank|
        # +----------+----------+--------------------+--------------------+
        # |        11|         1|               liked|              2.5165|
        # +----------+----------+--------------------+--------------------+
        # """
        # textrank_keywords_df = sentence_df.rdd.mapPartitions(textrank).toDF(
        #     ["article_id", "channel_id", "keyword", "textrank"])
        # textrank_keywords_df.write.insertInto("textrank_keywords_values")

        # return textrank_keywords_df, keywordsByTFIDF
        return keywordsByTFIDF

    # def get_article_profile(self, textrank, keywordsByTFIDF):
    #     """
    #     文章画像主题词建立
    #     :param idf: 所有词的idf值
    #     :param textrank: 每个文章的textrank值
    #     :return: 返回建立号增量文章画像
    #     article_profile：
    #     +----------+----------+--------------------------------------------+
    #     |article_id|channel_id|                         keywords|    topics|
    #     +----------+----------+--------------------+-----------------------+
    #     |    141462|         3|            {美丽:5.46, 心情:4.87}| [心情,美丽]|
    #     +----------+----------+-------------------------------------------+
    #     """
    #     keywordsByTFIDF = keywordsByTFIDF.withColumnRenamed("keyword", "keyword1")  # 将"keyword"字段重命名为"keyword1"
    #     # result在textrank_keywords_df的基础上增加了tfidf字段
    #     result = textrank.join(keywordsByTFIDF, textrank.keyword == keywordsByTFIDF.keyword1)  # 默认采用inner join
    #
    #     # 1、关键词（词，权重）
    #     # 计算关键词权重（withColumn方法新增属性"weights"）
    #     _articleKeywordsWeights = result.withColumn("weights", result.textrank * result.idf).select(
    #         ["article_id", "channel_id", "keyword", "weights"])
    #
    #     # 按article_id分组，合并关键词权重到字典；articleKeywordsWeights含(article_id、channel_id、keyword_list、weights_list)
    #     _articleKeywordsWeights.registerTempTable("temptable")
    #     articleKeywordsWeights = self.spark.sql(
    #         "select article_id, min(channel_id) channel_id, collect_list(keyword) keyword_list, collect_list(weights) weights_list from temptable group by article_id")
    #
    #     def _func(row):
    #         return row.article_id, row.channel_id, dict(zip(row.keyword_list, row.weights_list))
    #
    #     articleKeywords = articleKeywordsWeights.rdd.map(_func).toDF(["article_id", "channel_id", "keywords"])
    #
    #     # 2、主题词
    #     # 将tfidf和textrank共现的词作为主题词
    #     topic_sql = """
    #             select t.article_id article_id2, collect_set(t.keyword) topics from tfidf_keywords_values t
    #             inner join
    #             textrank_keywords_values r
    #             where t.keyword=r.keyword
    #             group by article_id2
    #             """
    #     articleTopics = self.spark.sql(topic_sql)
    #
    #     # 3、将主题词表和关键词表进行合并，插入表
    #     articleProfile = articleKeywords.join(articleTopics,
    #                                           articleKeywords.article_id == articleTopics.article_id2).select(
    #         ["article_id", "channel_id", "keywords", "topics"])
    #     articleProfile.write.insertInto("article_profile")
    #
    #     del keywordsByTFIDF
    #     del _articleKeywordsWeights
    #     del articleKeywords
    #     del articleTopics
    #     gc.collect()
    #
    #     return articleProfile
    def get_article_profile(self, keywordsByTFIDF):
        """
        文章画像主题词建立
        :param idf: 所有词的idf值
        :param textrank: 每个文章的textrank值
        :return: 返回建立号增量文章画像
        article_profile：
        +----------+----------+--------------------------------------------+
        |article_id|channel_id|                         keywords|    topics|
        +----------+----------+--------------------+-----------------------+
        |    141462|         3|            {美丽:5.46, 心情:4.87}| [心情,美丽]|
        +----------+----------+-------------------------------------------+
        """
        # 1、关键词（词，权重）
        # 计算关键词权重（withColumn方法新增属性"weights"）
        _articleKeywordsWeights = keywordsByTFIDF.withColumnRenamed("tfidf", "weights").select(
            ["article_id", "channel_id", "keyword", "weights"])

        # 按article_id分组，合并关键词权重到字典；articleKeywordsWeights含(article_id、channel_id、keyword_list、weights_list)
        _articleKeywordsWeights.registerTempTable("temptable")
        articleKeywordsWeights = self.spark.sql(
            "select article_id, min(channel_id) channel_id, collect_list(keyword) keyword_list, collect_list(weights) weights_list from temptable group by article_id")

        def _func(row):
            return row.article_id, row.channel_id, dict(zip(row.keyword_list, row.weights_list))

        articleKeywords = articleKeywordsWeights.rdd.map(_func).toDF(["article_id", "channel_id", "keywords"])

        # 2、主题词
        # 将tfidf和textrank共现的词作为主题词
        topic_sql = """
                select article_id article_id2, collect_set(keyword) topics from tfidf_keywords_values group by article_id2
                """
        articleTopics = self.spark.sql(topic_sql)

        # 3、将主题词表和关键词表进行合并，插入表
        articleProfile = articleKeywords.join(articleTopics,
                                              articleKeywords.article_id == articleTopics.article_id2).select(
            ["article_id", "channel_id", "keywords", "topics"])
        articleProfile.write.insertInto("article_profile")

        del keywordsByTFIDF
        del _articleKeywordsWeights
        del articleKeywords
        del articleTopics
        gc.collect()

        return articleProfile


if __name__ == '__main__':
    ua = UpdateArticle()
    # 获取增量文章数据
    sentence_df = ua.merge_article_data()
    if sentence_df.rdd.collect():
        # rank, idf = ua.generate_article_label(sentence_df)
        # articleProfile = ua.get_article_profile(rank, idf)
        idf = ua.generate_article_label(sentence_df)
        articleProfile = ua.get_article_profile(idf)



