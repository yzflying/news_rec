import os
import sys
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.join(BASE_DIR))

PYSPARK_PYTHON = "/miniconda2/envs/reco_sys/bin/python"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON

from offline import SparkSessionBase


class UpdateRecall(SparkSessionBase):
    SPARK_APP_NAME = "updateRecall"
    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


ur = UpdateRecall()

"""将 user_article_basic 表的clicked属性更改为0、1数据类型"""
def change_types(row):
    return row.user_id, row.article_id, int(row.clicked)

ur.spark.sql("use profile")
user_article_click = ur.spark.sql("select * from user_article_basic").select(['user_id', 'article_id', 'clicked'])
user_article_click = user_article_click.rdd.map(change_types).toDF(['user_id', 'article_id', 'clicked'])


"""用户ID与文章ID处理，编程ID索引（对user_id、user_id样本进行编号）"""
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
# 用户和文章ID超过ALS最大整数值，需要使用StringIndexer进行转换
user_id_indexer = StringIndexer(inputCol='user_id', outputCol='als_user_id')
article_id_indexer = StringIndexer(inputCol='user_id', outputCol='als_article_id')
pip = Pipeline(stages=[user_id_indexer, article_id_indexer])
pip_fit = pip.fit(user_article_click)
als_user_article_click = pip_fit.transform(user_article_click)


"""ALS协同过滤推荐算法，基于als_user_id、als_article_id、clicked列得到结果(二维数组)"""
from pyspark.ml.recommendation import ALS
# 模型训练和推荐默认每个用户固定文章个数
als = ALS(userCol='als_user_id', itemCol='als_article_id', ratingCol='clicked', checkpointInterval=1)
model = als.fit(als_user_article_click)
# recall_res含两列，als_user_id、recommendations(长度为100)
recall_res = model.recommendForAllUsers(100)    # 对每个user推荐100个item


# recall_res得到需要使用StringIndexer变换后的下标
# 保存原来的下表映射关系
refection_user = als_user_article_click.groupBy(['user_id']).max('als_user_id').withColumnRenamed('max(als_user_id)', 'als_user_id')
refection_article = als_user_article_click.groupBy(['article_id']).max('als_article_id').withColumnRenamed('max(als_article_id)', 'als_article_id')

"""
refection_user映射关系表,recommendations列表每个元素即文章编号及其推荐值
recall_res = recall_res.join(refection_user)，结果如下：
+----------  -+----------------- ----  ---+-------------- -----+
| als_user_id | recommendations           |         user_id    |
+--------  ---+---------------------------+----------------- --+
| 8           |[[163, 0.91],[163, 0.91]...|                   2|
| 0           |[[145, 0.653115],      ... | 1106476833370537984|
"""
recall_res = recall_res.join(refection_user, on=['als_user_id'], how='left').select(['als_user_id', 'recommendations', 'user_id'])

"""
recall_res = recall_res.withColumn('als_article_id')结果如下：
+---------  --+-------+----------------+
| als_user_id |user_id| als_article_id |
+---------  --+-------+----------------+
| 8           | 2     | [163, 0.913281]|
| 8           | 2     | [132, 0.913281]|
"""
import pyspark.sql.functions as F
recall_res = recall_res.withColumn('als_article_id', F.explode('recommendations')).drop('recommendations')


def _article_id(row):
  return row.als_user_id, row.user_id, row.als_article_id[0]

"""
als_recall = recall_res.rdd.map(_article_id)，结果如下：进行索引对应文章ID获取
+---------  --+-------+----------------+
| als_user_id |user_id| als_article_id |
+---------  --+-------+----------------+
| 8           | 2     |             163|
| 8           | 2     |             132|
"""
als_recall = recall_res.rdd.map(_article_id).toDF(['als_user_id', 'user_id', 'als_article_id'])
"""
als_recall = als_recall.join(refection_article)，结果如下：
+-------+----------------+
|user_id|     article_id |
+-------+----------------+
| 2     |              63|
| 2     |              32|
"""
als_recall = als_recall.join(refection_article, on=['als_article_id'], how='left').select(['user_id', 'article_id'])


"""获取每个文章对应的频道，推荐给用户时按照频道存储"""
ur.spark.sql("use toutiao")
news_article_basic = ur.spark.sql("select article_id, channel_id from news_article_basic")
# 最终含user_id、channel_id、article_list三个属性
als_recall = als_recall.join(news_article_basic, on=['article_id'], how='left')
als_recall = als_recall.groupBy(['user_id', 'channel_id']).agg(F.collect_list('article_id')).withColumnRenamed('collect_list(article_id)', 'article_list')
als_recall = als_recall.dropna()    # 删除推荐文章可能为空的channel


def save_offline_recall_hbase(partition):
    """离线模型召回结果存储"""
    import happybase
    pool = happybase.ConnectionPool(size=10, host='hadoop-master', port=9090)
    for row in partition:
        with pool.connection() as conn:
            # 获取历史看过的该频道文章
            history_table = conn.table('history_recall')
            # 多个版本
            data = history_table.cells('reco:his:{}'.format(row.user_id).encode(),
                                       'channel:{}'.format(row.channel_id).encode())

            history = []
            if len(data) >= 2:
                for l in data[:-1]:
                    history.extend(eval(l))
            else:
                history = []

            # 过滤reco_article与history
            reco_res = list(set(row.article_list) - set(history))

            if reco_res:
                table = conn.table('cb_recall')
                # 默认放在推荐频道
                table.put('recall:user:{}'.format(row.user_id).encode(),
                          {'als:{}'.format(row.channel_id).encode(): str(reco_res).encode()})
                conn.close()

                # 放入历史推荐过文章
                history_table.put("reco:his:{}".format(row.user_id).encode(),
                                  {'channel:{}'.format(row.channel_id): str(reco_res).encode()})
            conn.close()


als_recall.foreachPartition(save_offline_recall_hbase)


"""基于内容召回"""
# 过滤掉用户已经点击过的文章
ur.spark.sql("use profile")
user_article_basic = ur.spark.sql("select * from user_article_basic")
user_article_basic = user_article_basic.filter('clicked=True')


# 用户每次操作文章进行相似获取并进行推荐
# 基于内容相似召回（画像召回）
def save_content_filter_history_to__recall(partition):
    """计算每个用户的每个操作文章的相似文章，过滤之后，写入content召回表当中（支持不同时间戳版本）
    """
    import happybase
    pool = happybase.ConnectionPool(size=10, host='hadoop-master')

    # 进行为相似文章获取
    with pool.connection() as conn:

        # key:   article_id,    column:  similar:article_id
        similar_table = conn.table('article_similar')
        # 循环partition
        for row in partition:
            # 获取相似文章结果表
            similar_article = similar_table.row(str(row.article_id).encode(),
                                                columns=[b'similar'])
            # 相似文章相似度排序过滤，召回不需要太大的数据， 百个，千
            _srt = sorted(similar_article.items(), key=lambda item: item[1], reverse=True)
            if _srt:
                # 每次行为推荐10篇文章
                reco_article = [int(i[0].split(b':')[1]) for i in _srt][:10]

                # 获取历史看过的该频道文章
                history_table = conn.table('history_recall')
                # 多个版本
                data = history_table.cells('reco:his:{}'.format(row.user_id).encode(),
                                           'channel:{}'.format(row.channel_id).encode())

                history = []
                if len(data) >= 2:
                    for l in data[:-1]:
                        history.extend(eval(l))
                else:
                    history = []

                # 过滤reco_article与history
                reco_res = list(set(reco_article) - set(history))

                # 进行推荐，放入基于内容的召回表当中以及历史看过的文章表当中
                if reco_res:
                    # content_table = conn.table('cb_content_recall')
                    content_table = conn.table('cb_recall')
                    content_table.put("recall:user:{}".format(row.user_id).encode(),
                                      {'content:{}'.format(row.channel_id).encode(): str(reco_res).encode()})

                    # 放入历史推荐过文章
                    history_table.put("reco:his:{}".format(row.user_id).encode(),
                                      {'channel:{}'.format(row.channel_id).encode(): str(reco_res).encode()})

        conn.close()


user_article_basic.foreachPartition(save_content_filter_history_to__recall)
