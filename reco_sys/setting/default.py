channelInfo = {
            1: "html",
            2: "开发者资讯",
            3: "ios",
            4: "c++",
            5: "android",
            6: "css",
            7: "数据库",
            8: "区块链",
            9: "go",
            10: "产品",
            11: "后端",
            12: "linux",
            13: "人工智能",
            14: "php",
            15: "javascript",
            16: "架构",
            17: "前端",
            18: "python",
            19: "java",
            20: "算法",
            21: "面试",
            22: "科技动态",
            23: "js",
            24: "设计",
            25: "数码产品",
        }


# 增加spark online 启动配置
class DefaultConfig(object):
    """默认的一些配置信息"""
    # KAFKA配置
    KAFKA_SERVER = "192.168.19.137:9092"

    SPARK_ONLINE_CONFIG = (
        ("spark.app.name", "onlineUpdate"),  # 设置启动的spark的app名称，没有提供，将随机产生一个名称
        ("spark.master", "yarn"),
        ("spark.executor.instances", 4)
    )

# 增加redis配置信息，热门文章、新文章推荐使用


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
