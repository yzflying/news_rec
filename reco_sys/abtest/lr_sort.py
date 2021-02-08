import os
import sys
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))

PYSPARK_PYTHON = "/miniconda2/envs/reco_sys/bin/python"
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
from pyspark import SparkConf
from pyspark.sql import SparkSession
from utils import HBaseUtils
from server import pool
from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegressionModel
import pandas as pd


conf = SparkConf()
config = (
    ("spark.app.name", "sort"),
    ("spark.executor.memory", "2g"),    # 设置该app启动时占用的内存用量，默认1g
    ("spark.master", 'yarn'),
    ("spark.executor.cores", "2"),   # 设置spark executor使用的CPU核心数
)

conf.setAll(config)
spark = SparkSession.builder.config(conf=conf).getOrCreate()


# 1、读取用户特征中心特征
hbu = HBaseUtils(pool)
# 排序
# 读取特定用户特定频道的用户特征
try:
    user_feature = eval(hbu.get_table_row('ctr_feature_user', '{}'.format(1115629498121846784).encode(), 'channel:{}'.format(18).encode()))
except Exception as e:
    user_feature = []


# 2、读取文章特征中心特征，构造训练样本
if user_feature:

    result = []
    # 对召回结果列表中的每一篇文章id，读取文章特征
    for article_id in [17749, 17748, 44371, 44368]:
        try:
            article_feature = eval(hbu.get_table_row('ctr_feature_article',
                                                     '{}'.format(article_id).encode(),
                                                     'article:{}'.format(article_id).encode()))
        except Exception as e:
            article_feature = [0.0] * 111
        f = []
        # 第一个channel_id
        f.extend([article_feature[0]])
        # 第二个article_vector
        f.extend(article_feature[11:])
        # 第三个用户权重特征
        f.extend(user_feature)
        # 第四个文章权重特征
        f.extend(article_feature[1:11])
        vector = DenseVector(f)
        # result包含user_id、article_id、和特征vector(含channel_id、article_vector、user_feature、article_weights)
        result.append([1115629498121846784, article_id, vector])


# 3、加载数据和模型(离线排序训练得到的模型)
df = pd.DataFrame(result, columns=["user_id", "article_id", "features"])
test = spark.createDataFrame(df)
# 加载逻辑回归模型
model = LogisticRegressionModel.load("hdfs://hadoop-master:9000/headlines/models/LR.obj")
predict = model.transform(test)


def vector_to_double(row):
    return float(row.article_id), float(row.probability[1])


res = predict.select(['article_id', 'probability']).rdd.map(vector_to_double).toDF(['article_id', 'probability']).sort('probability', ascending=False)
# 排序后，只将排名在前100个文章ID返回给用户推荐
article_list = [i.article_id for i in res.collect()]
if len(article_list) > 100:
    article_list = article_list[:100]
reco_set = list(map(int, article_list))

