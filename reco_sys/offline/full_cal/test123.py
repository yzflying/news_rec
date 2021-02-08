import pandas as pd
from datetime import datetime
import time


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
            print('add', d, _localions)
    except Exception as e:
        # 已经关联过的异常忽略,partition与hdfs文件不直接关联
        pass