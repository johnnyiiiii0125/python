import logging
import requests_query
import os
import file
import db
from common import constant, utils


logger = logging.getLogger('cnn')
logging.basicConfig(level=logging.INFO)  # 设置日志级别


def download_cnn_news_list():
    keyword = 'olympic tokyo'
    page = 1
    while True:
        from_no = (page - 1) * constant.SIZE
        data = requests_query.api_search_cnn_news(keyword, page=page, cnn_from=from_no)
        if not data['result']:
            break
        page = page + 1
    logger.info('list cnn news end.')


def cnn_news_list_to_db():
    parent_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_CNN, constant.DIR_LIST)
    list_files = file.list_dirs(parent_dir)
    db_obj = db.DB()
    for file_name in list_files:
        logger.info("filename:" + file_name)
        if '.json' not in file_name:
            continue
        json_data = file.read_file_to_json(parent_dir, file_name)
        data_list = get_cnn_data_list(json_data)
        db_obj.insert_news(data_list)
    db_obj.close_conn()


def get_cnn_data_list(json_data):
    data_list = []
    for result_item in json_data['result']:
        # process date from string to datetime 2021-07-17T04:55:52Z
        date = utils.date_string_to_datetime(result_item['firstPublishDate'], '%Y-%m-%dT%H:%M:%S.%fZ')
        item = {'content': result_item['body'], 'url': result_item['url'], 'title': result_item['headline'],
                'date': date, 'source': 'cnn'}
        data_list.append(item)
    return data_list


download_cnn_news_list()
cnn_news_list_to_db()