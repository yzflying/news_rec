# 结果文件保存mysql数据库信息
mysql_db = 'mysql'   # 数据库
mysql_host = '10.201.2.209'
mysql_port = '3306'
mysql_user = 'root'
mysql_password = 'mysqladmin'


import pymysql


class mysql_tool:
    def __init__(self, _host, user_name, pwd, dbase, port=3306):
        self.host = _host
        self.user_name = user_name
        self.password = pwd
        self.database = dbase
        self.port = port

        try:
            self.conn = pymysql.connect(host=self.host, user=self.user_name,
                                        password=self.password, database=self.database, port=self.port, charset='utf8')
            self.cursor = self.conn.cursor()
        except Exception as err:
            print("Data Operation initialization failed: %s" % err)
            raise err

    def search_tool(self, sql):
        try:
            print(sql)
            self.cursor.execute(sql)  # 执行sql语句
            results = self.cursor.fetchall()  # 获取查询的所有记录
            print('ok')
            # 关闭
            self.cursor.close()
            self.conn.close()
        except Exception as e:
            raise e
        return results

    # 批量插入数据
    def batch_insert_tool(self, sql, data):
        try:
            self.cursor.executemany(sql, data)  # 执行sql语句
            self.conn.commit()
            print('ok')
            # 关闭
            self.cursor.close()
            self.conn.close()
        except Exception as e:
            print(e)


# if __name__ == '__main__':
#     my_model = mysql_tool(mysql_host, mysql_user, mysql_password, mysql_db, int(mysql_port))
#     data = [(165, 166, 0.3), (165, 167, 0.4)]
#     sql = "INSERT ignore INTO " + 'news_article_similar' + " (article_id, similar, EuclideanDistance)" + " VALUES ('%s', '%s', '%s') "
#     print(sql)
#     res = my_model.insert_tool(sql, data)
#     print(res)
