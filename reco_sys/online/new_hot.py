# 添加sparkstreaming启动对接kafka的配置

# new-article，新文章的读取  KAFKA配置
NEW_ARTICLE_DS = KafkaUtils.createDirectStream(stream_c, ['new-article'], click_kafkaParams)

from online import HOT_DS, NEW_ARTICLE_DS


class OnlineRecall(object):
    """实时处理（流式计算）部分"""
    def __init__(self):
        self.client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST, port=DefaultConfig.REDIS_PORT, db=10)
        # 在线召回筛选TOP-k个结果
        self.k = 20

    def _update_hot_redis(self):
        """更新热门文章  click-trace
        :return:
        """
        client = self.client

        def updateHotArt(rdd):
            for row in rdd.collect():
                logger.info("{}, INFO: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), row))
                # 如果是曝光参数，和阅读时长选择过滤
                if row['param']['action'] == 'exposure' or row['param']['action'] == 'read':
                    pass
                else:
                    # 解析每条行为日志，然后进行分析保存点击，喜欢，分享次数，这里所有行为都自增1
                    client.zincrby("ch:{}:hot".format(row['channelId']), 1, row['param']['articleId'])

        HOT_DS.map(lambda x: json.loads(x[1])).foreachRDD(updateHotArt)

    def _update_new_redis(self):
        """更新频道新文章 new-article
        :return:
        """
        client = self.client

        def computeFunction(rdd):
            for row in rdd.collect():
                channel_id, article_id = row.split(',')
                logger.info("{}, INFO: get kafka new_article each data:channel_id{}, article_id{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), channel_id, article_id))
                client.zadd("ch:{}:new".format(channel_id), {article_id: time.time()})

        NEW_ARTICLE_DS.map(lambda x: x[1]).foreachRDD(computeFunction)


from kafka import KafkaClient
client = KafkaClient(hosts="127.0.0.1:9092")
for topic in client.topics:
    print(topic)

from kafka import KafkaProducer

# kafka消息生产者
kafka_producer = KafkaProducer(bootstrap_servers=['192.168.19.137:9092'])
# 构造消息并发送
msg = '{},{}'.format(18, 13891)
kafka_producer.send('new-article', msg.encode())


