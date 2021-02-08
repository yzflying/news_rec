time=`date +"%Y-%m-%d" -d "-1day"`
declare -A check
check=([news_channel]=update_time)
declare -A merge
merge=([news_channel]=channel_id)

for k in ${!check[@]}
do
    sqoop import \
        --connect jdbc:mysql://10.201.2.209/mysql \
        --username root \
        --password mysqladmin \
        --table $k \
        --m 4 \
        --target-dir /user/hive/warehouse/di_news_db_test.db/$k \
        --fields-terminated-by "," \
        --incremental lastmodified \
        --check-column ${check[$k]} \
        --merge-key ${merge[$k]} \
        --last-value ${time}
done
