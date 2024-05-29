import pymysql
from common import constant

host = 'localhost'
user = 'root'
password = '123456'
db = 'fnews'
table_news = 'news'
table_news_urls = 'news_urls'
table_failed_news_urls = 'failed_news_urls'


class DB:
    def __init__(self):
        self.conn = None

    def get_conn(self):
        if not self.conn:
            self.conn = pymysql.connect(host=host, user=user, password=password, database=db)

    def insert_news(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_news + '(source, url, title, content, date) values (%s, %s, %s, %s, %s)'
                for item in data_list:
                    cursor.execute(sql, (item['source'], item['url'], item['title'], item['content'], item['date']))
            self.conn.commit()

    def insert_news_urls(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_news_urls + '(source, url) values (%s, %s)'
                for item in data_list:
                    cursor.execute(sql, (item['source'], item['url']))
            self.conn.commit()

    def insert_failed_news_urls(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'insert into ' + table_failed_news_urls + '(source, url) values (%s, %s)'
                for item in data_list:
                    cursor.execute(sql, (item['source'], item['url']))
            self.conn.commit()

    def update_news_date(self, data_list):
        self.get_conn()
        if self.conn:
            with self.conn.cursor() as cursor:
                sql = 'update ' + table_news + ' SET date=%s WHERE id=%s'
                for item in data_list:
                    cursor.execute(sql, (item['date'], item['id']))
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

    def query_news(self, page=None, where=None):
        sql = "select * from " + table_news
        if where is not None:
            sql += " " + where
        if page is not None:
            sql += " LIMIT " + str(constant.LIMIT) \
              + " OFFSET " + str((page - 1) * constant.LIMIT)
        result = self.query_results(sql)
        data = list(result['data'])
        news = []
        for item in data:
            news_item = {'id': item[0], 'source': item[1], 'url': item[2], 'title': item[3], 'content': item[4], 'date': item[5], 'updated': item[6]}
            news.append(news_item)
        return news

    def close_conn(self):
        if self.conn:
            self.conn.close()
            self.conn = None