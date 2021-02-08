import os
import sys
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.join(BASE_DIR))

PYSPARK_PYTHON = "/miniconda2/envs/reco_sys/bin/python"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from offline import SparkSessionBase


def _create_spark_hbase(self):
    conf = SparkConf()  # 创建spark config对象
    config = (
        ("spark.app.name", self.SPARK_APP_NAME),  # 设置启动的spark的app名称，没有提供，将随机产生一个名称
        ("spark.executor.memory", self.SPARK_EXECUTOR_MEMORY),  # 设置该app启动时占用的内存用量，默认2g
        ("spark.master", self.SPARK_URL),  # spark master的地址
        ("spark.executor.cores", self.SPARK_EXECUTOR_CORES),  # 设置spark executor使用的CPU核心数，默认是1核心
        ("spark.executor.instances", self.SPARK_EXECUTOR_INSTANCES),
        ("hbase.zookeeper.quorum", "192.168.19.137"),     # 新增
        ("hbase.zookeeper.property.clientPort", "22181")  # 新增
    )
    conf.setAll(config)
    # 利用config对象，创建spark session
    if self.ENABLE_HIVE_SUPPORT:
        return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    else:
        return SparkSession.builder.config(conf=conf).getOrCreate()


class CtrLogisticRegression(SparkSessionBase):
    SPARK_APP_NAME = "ctrLogisticRegression"
    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = _create_spark_hbase(self)


ctr = CtrLogisticRegression()


"""
用户行为日志读取,得到 user_article_basic
+-------------------+----------+----------+-------+
|            user_id|article_id|channel_id|clicked|
+-------------------+----------+----------+-------+
|1105045287866466304|     14225|         0|  false|
"""
ctr.spark.sql("use profile")
user_article_basic = ctr.spark.sql("select * from user_article_basic").select(['user_id', 'article_id', 'channel_id', 'clicked'])

"""
用户画像读取，并与行为日志合并, user_profile_hbase
+--------------------+--------+------+--------------------+
|             user_id|birthday|gender|     article_partial|
+--------------------+--------+------+--------------------+
|              user:1|     0.0|  null|Map(18:Animal -> ...|
"""
user_profile_hbase = ctr.spark.sql("select user_id, information.birthday, information.gender, article_partial, env from user_profile_hbase")
user_profile_hbase = user_profile_hbase.drop('env')

# "weights"字段的键值对分别是 {channel_id}:{topic}，weights
_schema = StructType([
    StructField("user_id", LongType()),
    StructField("birthday", DoubleType()),
    StructField("gender", BooleanType()),
    StructField("weights", MapType(StringType(), DoubleType()))
])

def get_user_id(row):
    return int(row.user_id.split(":")[1]), row.birthday, row.gender, row.article_partial

user_profile_hbase_temp = user_profile_hbase.rdd.map(get_user_id)
user_profile_hbase_schema = ctr.spark.createDataFrame(user_profile_hbase_temp, schema=_schema)

"""
train:
+-------------------+----------+-------+--------+------+--------------------+
|            user_id|article_id|clicked|birthday|gender|             weights|
+-------------------+----------+-------+--------+------+--------------------+
|1106473203766657024|     13778|  false|     0.0|  null|Map(18:text -> 0....|
"""
train = user_article_basic.join(user_profile_hbase_schema, on=['user_id'], how='left').drop('channel_id')


ctr.spark.sql("use article")
article_vector = ctr.spark.sql("select * from article_vector")
"""
train:
+-------------------+-------------------+-------+--------------------+----------+--------------------+
|         article_id|            user_id|clicked|             weights|channel_id|       articlevector|
+-------------------+-------------------+-------+--------------------+----------+--------------------+
|              13401|                 10|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|
"""
# 删除缺失值较多的生日、性别字段
train = train.join(article_vector, on=['article_id'], how='left').drop('birthday').drop('gender')

"""
article_profile：
+----------+----------+--------------------------------------------+
|article_id|channel_id|                         keywords|    topics|
+----------+----------+--------------------+-----------------------+
|    141462|         3|            {美丽:5.46, 心情:4.87}| [心情,美丽]|
+----------+----------+-------------------------------------------+
"""
ctr.spark.sql("use article")
article_profile = ctr.spark.sql("select * from article_profile")


def article_profile_to_feature(row):
    try:
        weights = sorted(row.keywords.values())[:10]
    except Exception as e:
        weights = [0.0] * 10
    return row.article_id, weights


"""
article_profile：
+----------+----------  ---+
|article_id|article_weights|
+----------+-----     -----+
|    141462|   [5.46, 4.87]| 
+----------+------ --- ----+
"""
article_profile = article_profile.rdd.map(article_profile_to_feature).toDF(['article_id', 'article_weights'])
"""
train:
+-------------------+-------------------+-------+--------------------+----------+--------------------+----------  ---+
|         article_id|            user_id|clicked|             weights|channel_id|       articlevector|article_weights|
+-------------------+-------------------+-------+--------------------+----------+--------------------+-----     -----+
|              13401|                 10|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|   [5.46, 4.87]| 
"""
train = train.join(article_profile, on=['article_id'], how='left')
train = train.dropna()

columns = ['article_id', 'user_id', 'channel_id', 'articlevector', 'weights', 'article_weights', 'clicked']


def feature_preprocess(row):
    from pyspark.ml.linalg import Vectors
    try:
        weights = sorted([row.weights[key] for key in row.weights.keys() if key[:2] == str(row.channel_id)])[:10]
    except Exception:
        weights = [0.0] * 10

    return row.article_id, row.user_id, row.channel_id, Vectors.dense(row.articlevector), Vectors.dense(weights), Vectors.dense(article_weights), int(row.clicked)


# 对weights进行处理：如果channel是对应的channel_id，则保存相应的标签权重值
train = train.rdd.map(feature_preprocess).toDF(columns)


cols = ['article_id', 'user_id', 'channel_id', 'articlevector', 'weights', 'article_weights', 'clicked']

# 特征：channel_id、articlevector(100维度)、weights(10个)、article_weights(10个)
train_version_two = VectorAssembler().setInputCols(cols[2:6]).setOutputCol("features").transform(train)

lr = LogisticRegression()
model = lr.setLabelCol("clicked").setFeaturesCol("features").fit(train_version_two)
model.save("hdfs://hadoop-master:9000/headlines/models/lr.obj")

# 读取模型
online_model = LogisticRegressionModel.load("hdfs://hadoop-master:9000/headlines/models/CtrLogistic.obj")

res_transfrom = online_model.transform(train_version_two)
# probability[0]：不点击概率；probability[1]：点击概率
res_transfrom.select(["clicked", "probability", "prediction"]).show()


def vector_to_double(row):
    return float(row.clicked), float(row.probability[1])


score_label = res_transfrom.select(["clicked", "probability"]).rdd.map(vector_to_double)

