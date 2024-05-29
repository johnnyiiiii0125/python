import pymysql
import json

host = 'localhost'
user = 'root'
password = '123456'
db = 'weibo'
table_news = 'search_result2'
table_weibolist = 'weibo_list'
table_weibodetail = 'weibo_detail'
LIMIT = 20


class DB:
    def __init__(self):
        self.conn = None

    def get_conn(self):
        if not self.conn:
            self.conn = pymysql.connect(host=host, user=user, password=password, database=db)

    def insert_weibolist(self, data_list):
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_weibolist + '(weiboid, userid, username, keyword, mid) ' \
                                                    'values ' \
                                                    '(%s, %s, %s, %s, %s)'
                for item in data_list:
                    try:
                        cursor.execute(sql, (item['weiboid'], item['userid'], item['username'],
                                             item['keyword'], item['mid']))
                    except Exception as e:
                        print(e)
            self.conn.commit()

    def insert_detail(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_weibodetail + '(weiboid, userid, username, keyword, url, content, ' \
                                                           'time, forward_count, like_count, reply_count, mid) ' \
                                                    'values ' \
                                                    '(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                for item in data_list:
                    # try:
                    cursor.execute(sql, (item['weiboid'], item['userid'], item['username'], item['keyword'],
                                         item['url'], item['content'], item['time'],
                                         item['forward_count'], item['like_count'],
                                         item['reply_count'], item['mid']))
                    # except Exception as e:
                    #     print(e)
            self.conn.commit()

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

    def query_weibo_list(self, page):
        offset = (page - 1) * LIMIT
        sql = 'select * from ' + table_weibolist + ' where processed=0 and mid is not null ' \
                                                   'order by id asc limit ' + str(offset) + ',' + str(LIMIT)
        result = self.query_results(sql)
        return list(result['data'])

    def mark_list_as_processed(self, ids):
        self.get_conn()
        ids_str = []
        for id in ids:
            ids_str.append(str(id))
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'update ' + table_weibolist + ' SET processed=1 WHERE id in (' + ','.join(ids_str) + ')'
                cursor.execute(sql)
            self.conn.commit()

    def update_mid(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'update ' + table_weibolist + ' SET mid=%s WHERE weiboid=%s'
                for item in data_list:
                    cursor.execute(sql, (item['mid'], item['weiboid']))
                sql2 = 'update ' + table_weibodetail + ' SET mid=%s WHERE weiboid=%s'
                for item in data_list:
                    cursor.execute(sql2, (item['mid'], item['weiboid']))
            self.conn.commit()

    def close_conn(self):
        if self.conn:
            self.conn.close()
            self.conn = None