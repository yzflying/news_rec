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
import pyhdfs
import time


class UpdateUserProfile(SparkSessionBase):
    """离线相关处理程序
    """
    SPARK_APP_NAME = "updateUser"
    ENABLE_HIVE_SUPPORT = True

    SPARK_EXECUTOR_MEMORY = "7g"

    def __init__(self):
        self.spark = self._create_spark_session()

    def update_user_label(self, sqlDF):
        """
        如果user_action存在新的用户日志，本函数进行更新用户画像
        :param sqlDF: user_action
        :return: 将用户画像结果保存到 hbase表 user_profile
        """

        def _compute(row):
            # 进行判断行为类型
            _list = []
            if row.action == "exposure":
                for article_id in eval(row.articleId):
                    _list.append([row.userId, row.actionTime, article_id, row.channelId, False, False, False, True,
                                  row.readTime])
                return _list
            else:
                class Temp(object):
                    shared = False
                    clicked = False
                    collected = False
                    read_time = ""

                _tp = Temp()
                if row.action == "share":
                    _tp.shared = True
                elif row.action == "click":
                    _tp.clicked = True
                elif row.action == "collect":
                    _tp.collected = True
                elif row.action == "read":
                    _tp.clicked = True
                else:
                    pass
                _list.append([row.userId, row.actionTime, int(row.articleId), row.channelId, _tp.shared, _tp.clicked,
                              _tp.collected, True, row.readTime])
                return _list

        """
        sqlDF大致形式如下：
        {"actionTime":"2019-03-08 16:50:36",
        "readTime":"",
        "channelId":0,
        "action": "exposure", 
        "userId": "10", 
        "articleId": "[44386, 44739]", 
        "algorithmCombine": "C2"}}
        """
        _res = sqlDF.rdd.flatMap(_compute)
        data = _res.toDF(
            ["user_id", "action_time", "article_id", "channel_id", "shared", "clicked", "collected", "exposure",
             "read_time"])

        """
        将user_action日志初步处理结果 data 与先前的初步处理结果 user_article_basic 进行合并；
        按照 user_id, article_id分组，将分组结果更新到 user_article_basic
        """
        uup.spark.sql("use di_news_db_test")
        old = uup.spark.sql("select * from user_article_basic")
        # 由于合并的结果中不是对于user_id和article_id唯一的，一个用户会对文章多种操作
        new_old = old.unionAll(data)
        new_old.registerTempTable("temptable")
        # 按照用户，文章分组存放进去 user_article_basic（保存用户画像结果）各属性取较大值
        uup.spark.sql(
            "insert overwrite table user_article_basic select user_id, max(action_time) as action_time, "
            "article_id, max(channel_id) as channel_id, max(shared) as shared, max(clicked) as clicked, "
            "max(collected) as collected, max(exposure) as exposure, max(read_time) as read_time from temptable "
            "group by user_id, article_id")

        """获取更新后的用户日志初步处理结果 user_article_basic，然后与文章画像article_profile的联结，获取主题词"""
        uup.spark.sql("use profile")
        # 取出日志中的channel_id
        user_article_ = uup.spark.sql("select * from user_article_basic").drop('channel_id')
        """
        article_profile：
        +----------+----------+--------------------------------------------+
        |article_id|channel_id|                         keywords|    topics|
        +----------+----------+--------------------+-----------------------+
        |    141462|         3|            {美丽:5.46, 心情:4.87}| [心情,美丽]|
        +----------+----------+-------------------------------------------+
        """
        uup.spark.sql('use article')
        article_label = uup.spark.sql("select article_id, channel_id, topics from article_profile")
        # 合并使用文章中正确的channel_id
        click_article_res = user_article_.join(article_label, how='left', on=['article_id'])
        # 将topics字段的列表爆炸(topics字段的关键词列表长度为n，则爆炸后该样本变为n条样本；每个样本用户行为相关字段一致，仅topic字段为不同的、单个主题词)
        import pyspark.sql.functions as F
        # 新建topic列，删除topics列，将topics列的列表爆炸，每个列表元素作为topic属性的值
        click_article_res = click_article_res.withColumn('topic', F.explode('topics')).drop('topics')

        """
        # 对不同行为类型赋予一定的权重
        # 用户标签权重 =( 行为类型权重之和) × 时间衰减
        # 计算每个用户对每篇文章的标签的权重
        """
        def compute_weights(rowpartition):
            """处理每个用户对文章的点击数据"""
            weightsOfaction = {
                "read_min": 1,
                "read_middle": 2,
                "collect": 2,
                "share": 3,
                "click": 5
            }

            import happybase
            from datetime import datetime
            import numpy as np
            #  用于读取hbase缓存结果配置
            pool = happybase.ConnectionPool(size=10, host='192.168.19.137', port=9090)

            # 读取文章的标签数据
            # 计算权重值
            # 时间间隔
            for row in rowpartition:

                t = datetime.now() - datetime.strptime(row.action_time, '%Y-%m-%d %H:%M:%S')
                # 时间衰减系数
                time_exp = 1 / (np.log(t.days + 1) + 1)

                if row.read_time == '':
                    r_t = 0
                else:
                    r_t = int(row.read_time)
                # 浏览时间分数
                is_read = weightsOfaction['read_middle'] if r_t > 1000 else weightsOfaction['read_min']

                # 每个词的权重分数
                weigths = time_exp * (
                        row.shared * weightsOfaction['share'] + row.collected * weightsOfaction['collect'] + row.
                        clicked * weightsOfaction['click'] + is_read)

                with pool.connection() as conn:
                    table = conn.table('user_profile')
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'partial:{}:{}'.format(row.channel_id, row.topic).encode(): json.dumps(
                                  weigths).encode()})
                    conn.close()

        click_article_res.foreachPartition(compute_weights)

    def update_user_info(self):
        """将用户的基础信息（gender, age）添加到用户画像 user_profile"""
        self.spark.sql("use toutiao")
        user_basic = self.spark.sql("select user_id, gender, birthday from user_profile")

        # 更新用户基础信息
        def _udapte_user_basic(partition):
            """更新用户基本信息
            """
            import happybase
            #  用于读取hbase缓存结果配置
            pool = happybase.ConnectionPool(size=10, host='172.17.0.134', port=9090)
            for row in partition:
                from datetime import date

                age = 0
                if row.birthday != 'null':
                    born = datetime.strptime(row.birthday, '%Y-%m-%d')
                    today = date.today()
                    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))

                with pool.connection() as conn:
                    table = conn.table('user_profile')
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'basic:gender'.encode(): json.dumps(row.gender).encode()})
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'basic:birthday'.encode(): json.dumps(age).encode()})
                    conn.close()

        user_basic.foreachPartition(_udapte_user_basic)


uup = UpdateUserProfile()
"""
spark hive手动关联user_action表所有日期文件
在进行日志信息的处理之前，先将我们之前建立的user_action表之间进行所有日期关联，spark hive不会自动关联
"""
import pandas as pd
from datetime import datetime

def datelist(beginDate, endDate):
    date_list = [datetime.strftime(x, '%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_list

dl = datelist("2019-03-05", time.strftime("%Y-%m-%d", time.localtime()))

from hdfs.client import Client
fs = Client("http://10.201.2.211:9870")
for d in dl:
    try:
        _localions = '/user/hive/warehouse/di_news_db_test.db/user_action/' + d
        if d in fs.list('/user/hive/warehouse/di_news_db_test.db/user_action/'):
            uup.spark.sql("alter table user_action add partition (dt="+d+") location "+_localions)
    except Exception as e:
        # 已经关联过的异常忽略,partition与hdfs文件不直接关联
        pass


# """如果hadoop没有今天该日期文件，则没有日志数据，结束"""
# time_str = time.strftime("%Y-%m-%d", time.localtime())
# _localions = '/user/hive/warehouse/di_news_db_test.db/user_action/' + time_str
# if fs.exists(_localions):
#     # 如果有该文件直接关联，捕获关联重复异常
#     try:
#         uup.spark.sql("alter table user_action add partition (dt='%s') location '%s'" % (time_str, _localions))
#     except Exception as e:
#         pass
#
#     sqlDF = uup.spark.sql(
#                 "select actionTime, readTime, channelId, param.articleId, param.algorithmCombine, param.action, param.userId from user_action where dt={}".format(time_str))
# else:
#     pass
#
#
# # 选定某个日期的用户行为日志user_action，进行用户画像
# sqlDF = uup.spark.sql("select actionTime, readTime, channelId, param.articleId, param.algorithmCombine, param.action, param.userId from user_action where dt>='2018-01-01'")
#
# if sqlDF.collect():
#     uup.update_user_label(sqlDF)
#
#     """hbase查询结果"""
#     import happybase
#     #  用于读取hbase缓存结果配置
#     pool = happybase.ConnectionPool(size=10, host='192.168.19.137', port=9090)
#     with pool.connection() as conn:
#         table = conn.table('user_profile')
#         # 获取每个键 对应的所有列的结果
#         data = table.row(b'user:2', columns=[b'partial'])
#         conn.close()
#
# """更新用户基本信息"""
# uup.update_user_info()
