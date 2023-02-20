import pymysql

host = 'localhost'
user = 'root'
password = '123456'
db = 'fnews'
table_news = 'news'


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