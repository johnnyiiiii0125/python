import pymysql
import json

host = 'localhost'
user = 'root'
password = '123456'
db = 'weibo'
table_news = 'search_result2'


class DB:
    def __init__(self):
        self.conn = None

    def get_conn(self):
        if not self.conn:
            self.conn = pymysql.connect(host=host, user=user, password=password, database=db)

    def insert_results(self, data_list):
        # self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_news + '(keyword, poster_id, poster_name, poster_home, poster_verify, ' \
                                                    'post_content, post_url, post_time, post_device, ' \
                                                    'post_forward_count, post_reply_count, post_like_count, ' \
                                                    'post_original_url, post_media, post_year, post_month, post_day) ' \
                                                    'values ' \
                                                    '(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ' \
                                                    '%s, %s, %s)'
                for item in data_list:
                    cursor.execute(sql, (item['keyword'], item['poster_id'], item['poster_name'], item['poster_home'],
                                         item['poster_verify'], item['post_content'], item['post_url'],
                                         item['post_time'], item['post_device'], item['post_forward_count'],
                                         item['post_reply_count'], item['post_like_count'],
                                         item['post_original_url'], json.dumps(item['post_media']),
                                         item['post_year'], item['post_month'], item['post_day']))
            self.conn.commit()

    def query_results(self, sql):
        self.get_conn()
        result = {}
        if self.conn:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                data = cursor.fetchall()
                header = [t[0] for t in cursor.description]
                result['data'] = data
                result['header'] = header
            self.conn.commit()
        return result

    def close_conn(self):
        if self.conn:
            self.conn.close()
            self.conn = None