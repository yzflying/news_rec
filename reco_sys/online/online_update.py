import os
import sys
BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))
print(BASE_DIR)
PYSPARK_PYTHON = "/miniconda2/envs/reco_sys/bin/python"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.2 pyspark-shell"
from online import stream_sc, SIMILAR_DS, pool
from setting.default import DefaultConfig
from datetime import datetime
import setting.logging as lg
import logging
import redis
import json
import time


# 注意，如果是使用jupyter或ipython中，利用spark streaming链接kafka的话，必须加上下面语句
# 同时注意：spark version>2.2.2的话，pyspark中的kafka对应模块已被遗弃，因此这里暂时只能用2.2.2版本的spark
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.2 pyspark-shell"


class OnlineRecall(object):
    """
    在线处理计算平台
    1.在线内容召回，实时写入用户点击文章的相似文章
    2.在线新文章召回、在线热门文章召回
    """
    def __init__(self):
        pass

    def _update_online_cb(self):
        """
        在线内容召回计算
        通过点击行为更新用户的cb召回表中的online召回结果
        :return:
        """
        def foreachFunc(rdd):
            for data in rdd.collect():
                # 判断日志行为类型，只处理点击流日志
                if data["param"]["action"] in ["click", "collect", "share"]:
                    # print(data)
                    with pool.connection() as conn:
                        try:
                            # 相似文章表
                            sim_table = conn.table("article_similar")
                            # 根据用户点击流日志涉及文章找出与之最相似文章(基于内容的相似)，选取TOP-k相似的作为召回推荐结果
                            _dic = sim_table.row(str(data["param"]["articleId"]).encode(), columns=[b"similar"])
                            _srt = sorted(_dic.items(), key=lambda obj: obj[1], reverse=True)  # 按相似度排序
                            if _srt:

                                topKSimIds = [int(i[0].split(b":")[1]) for i in _srt[:self.k]]

                                # 根据历史推荐集过滤，已经给用户推荐过的文章
                                history_table = conn.table("history_recall")

                                _history_data = history_table.cells(
                                    b"reco:his:%s" % data["param"]["userId"].encode(),
                                    b"channel:%d" % data["channelId"])
                                # print("_history_data: ", _history_data)

                                history = []
                                if len(data) >= 2:
                                    for l in data[:-1]:
                                        history.extend(eval(l))
                                else:
                                    history = []

                                # 根据历史召回记录，过滤召回结果
                                recall_list = list(set(topKSimIds) - set(history_data))

                                # print("recall_list: ", recall_list)
                                logger.info("{}, INFO: store user:{} cb_recall data".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["param"]["userId"]))
                                if recall_list:
                                    # 如果有推荐结果集，那么将数据添加到cb_recall表中，同时记录到历史记录表中
                                    logger.info(
                                        "{}, INFO: get online-recall data".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                                    recall_table = conn.table("cb_recall")

                                    recall_table.put(
                                        b"recall:user:%s" % data["param"]["userId"].encode(),
                                        {b"online:%d" % data["channelId"]: str(recall_list).encode()}
                                    )

                                    history_table.put(
                                        b"reco:his:%s" % data["param"]["userId"].encode(),
                                        {b"channel:%d" % data["channelId"]: str(recall_list).encode()}
                                    )
                        except Exception as e:
                            logger.info("{}, WARN: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
                        finally:
                            conn.close()

        SIMILAR_DS.map(lambda x: json.loads(x[1])).foreachRDD(foreachFunc)


if __name__ == '__main__':
    """基于用户点击情况，依据内容相似，实时推荐相似文章"""
    op = OnlineRecall()
    op._update_online_cb()
    stream_c.start()
    # 使用 ctrl+c 可以退出服务
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

















