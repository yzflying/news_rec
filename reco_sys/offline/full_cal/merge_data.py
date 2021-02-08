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


class OriginArticleData(SparkSessionBase):
    SPARK_APP_NAME = "mergeArticle"
    SPARK_URL = "yarn"
    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


oa = OriginArticleData()
oa.spark.sql("use di_news_db_test")

# news_article_basic 与 news_article_content 两表基于 article_id 内连接
basic_content = oa.spark.sql("select a.article_id, a.channel_id, a.title, b.content from news_article_basic a inner \
                            join news_article_content b on a.article_id=b.article_id where a.article_id=141469")


import pyspark.sql.functions as F
import gc


# 基于basic_content的临时表temparticle
basic_content.registerTempTable("temparticle")
# basic_content 与 news_channel 两表基于 channel_id 左连接（在basic_content基础上增加channel_name字段）
channel_basic_content = oa.spark.sql("select t.*, n.channel_name from temparticle t left \
                                        join news_channel n on t.channel_id=n.channel_id")

# 利用concat_ws方法，将多列数据合并为一个长文本内容sentence（频道，标题以及内容合并）
sentence_df = channel_basic_content.select("article_id", "channel_id", "channel_name", "title", "content",
                                           F.concat_ws(
                                             ",",
                                             channel_basic_content.channel_name,
                                             channel_basic_content.title,
                                             channel_basic_content.content
                                           ).alias("sentence")
                                          )

# 清除缓存
del basic_content
del channel_basic_content


gc.collect()

"""
将宽表数据写入article.article_data(含"article_id", "channel_id", "channel_name", "title", "article_content","sentence")
141462 3 ios test-20190316-115123 今天天气不错，心情很美丽！！！ ios,test-20190316-115123,今天天气不错，心情很美丽！！！
"""
sentence_df.write.insertInto("article_data")
