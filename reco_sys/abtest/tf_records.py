# tf_records文件存储示例

# 保存到TFRecords文件中
df = train_res.select(['user_id', 'article_id', 'clicked', 'features'])
df_array = df.collect()
import pandas as pd
df = pd.DataFrame(df_array)


import tensorflow as tf


def write_to_tfrecords(click_batch, feature_batch):
        """
        将数据存进tfrecords，方便管理每个样本的属性
        :param image_batch: 特征值
        :param label_batch: 目标值
        :return: None
        """
        # 1、构造tfrecords的存储实例
        writer = tf.python_io.TFRecordWriter("./train_ctr_201905.tfrecords")
        # 2、循环将每个样本写入到文件当中
        for i in range(len(click_batch)):

            click = click_batch[i]
            feature = feature_batch[i].tostring()

            # 绑定每个样本的属性
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[click])),
                "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature])),
            }))
            writer.write(example.SerializeToString())

        # 文件需要关闭
        writer.close()
        return None


# 开启会话打印内容
with tf.Session() as sess:
    # 创建线程协调器
    coord = tf.train.Coordinator()

    # 开启子线程去读取数据
    # 返回子线程实例
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 存入数据
    write_to_tfrecords(df.iloc[:, 2], df.iloc[:, 3])

    # 关闭子线程，回收
    coord.request_stop()
    coord.join(threads)