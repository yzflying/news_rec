def compute_article_similar(self, articleProfile):
    """
    计算增量文章与历史文章的相似度 word2vec
    :return:
    """

    # 得到要更新的新文章通道类别(不采用)
    # all_channel = set(articleProfile.rdd.map(lambda x: x.channel_id).collect())
    def avg(row):
        x = 0
        for v in row.vectors:
            x += v
        #  将平均向量作为article的向量
        return row.article_id, row.channel_id, x / len(row.vectors)

    for channel_id, channel_name in CHANNEL_INFO.items():

        profile = articleProfile.filter('channel_id = {}'.format(channel_id))
        wv_model = Word2VecModel.load(
            "hdfs://hadoop-master:9000/headlines/models/channel_%d_%s.word2vec" % (channel_id, channel_name))
        vectors = wv_model.getVectors()

        # 计算向量
        profile.registerTempTable("incremental")
        articleKeywordsWeights = ua.spark.sql(
            "select article_id, channel_id, keyword, weight from incremental LATERAL VIEW explode(keywords) AS keyword, weight where channel_id=%d" % channel_id)

        articleKeywordsWeightsAndVectors = articleKeywordsWeights.join(vectors,
                                                                       vectors.word == articleKeywordsWeights.keyword,
                                                                       "inner")
        articleKeywordVectors = articleKeywordsWeightsAndVectors.rdd.map(
            lambda r: (r.article_id, r.channel_id, r.keyword, r.weight * r.vector)).toDF(
            ["article_id", "channel_id", "keyword", "weightingVector"])

        articleKeywordVectors.registerTempTable("tempTable")
        articleVector = self.spark.sql(
            "select article_id, min(channel_id) channel_id, collect_set(weightingVector) vectors from tempTable group by article_id").rdd.map(
            avg).toDF(["article_id", "channel_id", "articleVector"])

        # 写入数据库
        def toArray(row):
            return row.article_id, row.channel_id, [float(i) for i in row.articleVector.toArray()]

        articleVector = articleVector.rdd.map(toArray).toDF(['article_id', 'channel_id', 'articleVector'])
        articleVector.write.insertInto("article_vector")

        import gc
        del wv_model
        del vectors
        del articleKeywordsWeights
        del articleKeywordsWeightsAndVectors
        del articleKeywordVectors
        gc.collect()

        # 得到历史数据, 转换成固定格式使用LSH进行求相似
        train = self.spark.sql("select * from article_vector where channel_id=%d" % channel_id)

        def _array_to_vector(row):
            return row.article_id, Vectors.dense(row.articleVector)

        train = train.rdd.map(_array_to_vector).toDF(['article_id', 'articleVector'])
        test = articleVector.rdd.map(_array_to_vector).toDF(['article_id', 'articleVector'])

        brp = BucketedRandomProjectionLSH(inputCol='articleVector', outputCol='hashes', seed=12345,
                                          bucketLength=1.0)
        model = brp.fit(train)
        similar = model.approxSimilarityJoin(test, train, 2.0, distCol='EuclideanDistance')

        def save_hbase(partition):
            import happybase
            for row in partition:
                pool = happybase.ConnectionPool(size=3, host='hadoop-master')
                # article_similar article_id similar:article_id sim
                with pool.connection() as conn:
                    table = connection.table("article_similar")
                    for row in partition:
                        if row.datasetA.article_id == row.datasetB.article_id:
                            pass
                        else:
                            table.put(str(row.datasetA.article_id).encode(),
                                      {b"similar:%d" % row.datasetB.article_id: b"%0.4f" % row.EuclideanDistance})
                    conn.close()

        similar.foreachPartition(save_hbase)


# 添加函数到update_article.py文件中，修改update更新代码
ua = UpdateArticle()
sentence_df = ua.merge_article_data()
if sentence_df.rdd.collect():
    rank, idf = ua.generate_article_label(sentence_df)
    articleProfile = ua.get_article_profile(rank, idf)
    ua.compute_article_similar(articleProfile)