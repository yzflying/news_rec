from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkSessionBase(object):

    SPARK_APP_NAME = None # APP的名字
    SPARK_URL = "yarn" # 启动运行方式，单机为local

    SPARK_EXECUTOR_MEMORY = "6g" # 执行内存
    SPARK_EXECUTOR_CORES = 4  # 每个EXECUTOR能够使用的CPU core的数量
    SPARK_EXECUTOR_INSTANCES = 2 # 最多能够同时启动的EXECUTOR的实例个数

    ENABLE_HIVE_SUPPORT = False   # 是否启动hive支持

    def _create_spark_session(self):
        """给spark程序创建初始化spark session"""
        # 1、创建配置
        conf = SparkConf()

        config = (
            ("spark.app.name", self.SPARK_APP_NAME),  # 设置启动的spark的app名称，没有提供，将随机产生一个名称
            ("spark.executor.memory", self.SPARK_EXECUTOR_MEMORY),  # 设置该app启动时占用的内存用量，默认2g
            ("spark.master", self.SPARK_URL),  # spark master的地址
            ("spark.executor.cores", self.SPARK_EXECUTOR_CORES),  # 设置spark executor使用的CPU核心数，默认是1核心
            ("spark.executor.instances", self.SPARK_EXECUTOR_INSTANCES)
        )

        conf.setAll(config)

        # 2、读取配置初始化
        # 如果过开启HIVE信息
        if self.ENABLE_HIVE_SUPPORT:
            return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        else:
            return SparkSession.builder.config(conf=conf).getOrCreate()


