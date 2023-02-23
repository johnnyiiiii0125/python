import db
import file
import os
import pandas as pd

class Excel:
    def __init__(self):
        self.db_obj = db.DB()
        self.header = ['id', '媒体', '新闻链接', '标题', '内容', '发布时间', '添加时间', '修改时间']

    def data_from_db_to_excel(self, sql, file_dir, file_name):
        result = self.db_obj.query_results(sql)
        data = result['data']
        if len(data) > 0:
            file.check_and_make_dirs(file_dir)
            df = pd.DataFrame(list(data), columns=self.header)
            df.to_excel(os.path.join(file_dir, file_name), index=False)
        self.db_obj.close_conn()

    def read_excel(self, file_dir, filename):
        df = pd.read_excel(os.path.join(file_dir, filename))
        return df

