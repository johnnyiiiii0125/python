import pandas as pd
import db
import constant
import os
from openpyxl import load_workbook


DIR = '/Users/wangzhongyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/Johnnyiiiii/西德尼/project/data/工匠精神'

HEADER = ['id', '关键词', '发布者编号', '发布者', '发布者主页', '微博认证', '内容', '链接', '发布时间', '设备', '转发数',
                  '回复数', '赞数', '被转发链接', '多媒体', '发布年份', '发布月份', '发布日']
PAGE_SIZE = 5000
COUNTRIES = ['德国', '日本', '瑞士', '意大利', '中国']
OTHER_COUNTRIES = ['德国', '日本', '瑞士', '意大利']
REGION = ['南方', '北方', '城市', '农村', '城镇', '乡村']
CELEBRITY = ['王一博', '易烊千玺', '邓伦', '刘昊然', '杨幂', '胡歌', '秦俊杰', '娱乐圈']
QIYE = ['品牌', '集团', '企业', '公司']


def get_data_from_db_by_date(db_obj, year, month):
    result = db_obj.query_results('select * from search_result2 where post_year=' + str(year) + ' and post_month=' + str(month))
    data = result['data']
    if len(data) > 0:
        df = pd.DataFrame(list(data), columns=HEADER)
        df.to_excel(DIR + '/' + str(year) + '_' + str(month) + '.xlsx', index=False)
    month += 1
    if month > 12:
        month = 1
        year += 1
    if not (year >= 2022 and month >= 6):
        get_data_from_db_by_date(db_obj, year, month)


def preprocess_data(data, added_value):
    new_data = []
    for item in data:
        new_data.append(item + (added_value,))
    return new_data


def get_data_by_filters(db_obj, filters, file_name, filter_type):
    file_path = DIR + '/' + file_name + '.xlsx'
    sql_filter = ''
    for i in range(0, len(filters)):
        filter = filters[i]
        if i != 0:
            sql_filter = sql_filter + ' or '
        sql_filter = sql_filter + 'post_content like "%' + filter + '%"'
    page = 1
    while True:
        sql = 'select * from search_result where ' + sql_filter + ' limit '\
              + str((page-1) * PAGE_SIZE) + ', ' + str(PAGE_SIZE)
        result = db_obj.query_results(sql)
        data = result['data']
        if len(data) == 0:
            break
        page = page + 1
        header = HEADER.copy()
        # header.append(filter_type)
        df = pd.DataFrame(list(data), columns=header)
        if not os.path.exists(file_path):
            df.to_excel(file_path, index=False)
        else:
            df_old = pd.DataFrame(pd.read_excel(file_path))  # 读取原数据文件和表
            row_old = df_old.shape[0]  # 获取原数据的行数
            print("old_row" + str(row_old))
            book = load_workbook(file_path)
            writer = pd.ExcelWriter(file_path, engine='openpyxl')
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            df.to_excel(writer, index=False, header=False, startrow=row_old+1)
            writer.save()


def reprocess_country_file():
    file_name = '国家层面（中国、德国、日本、意大利）.xlsx'
    new_file_name = '国家层面（德国、日本、意大利）.xlsx'
    file_path = os.path.join(DIR, file_name)
    df = pd.read_excel(file_path)
    new_col = '通过'
    # 最后插入一列
    df[new_col] = '否'
    columns = df.columns.values.tolist()
    for idx, row in df.iterrows():
        content = row['内容']
        for item in OTHER_COUNTRIES:
            if item in content:
                df.loc[idx, new_col] = '是'
                break
    df.to_excel(os.path.join(DIR, new_file_name), index=False)


DETAIL_HEADER = ['发布者', '检索关键词', '微博链接', '内容', '发布时间', '转发数', '回复数', '赞数']
def detail_to_excel_by_username(db_obj):
    df = pd.read_excel(os.path.join(constant.ROOT_DIR, constant.GOV_WEIBO_FILENAME))
    for idx, row in df.iterrows():
        username = row['name']
        sql = 'select username, keyword, url, content, time, forward_count, reply_count, like_count ' \
              'from weibo_detail where username="' + username + '" order by id asc'
        result = db_obj.query_results(sql)
        data = result['data']
        if len(data) > 0:
            df = pd.DataFrame(list(data), columns=DETAIL_HEADER)
            df.to_excel(os.path.join(constant.ROOT_DIR, constant.ZHEJIANG_DIR, username + '-原始数据.xlsx'), index=False)



# reprocess_country_file()

db_obj = db.DB()
# get_data_from_db_by_date(db_obj, 2017, 3)
# get_data_by_filters(db_obj, QIYE, '企业', '企业')
# db_obj.close_conn()
detail_to_excel_by_username(db_obj)

