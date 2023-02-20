import selenium_query
from common import constant, utils
import os
import file
import datetime
import logging
import db


logger = logging.getLogger('bbc')
logging.basicConfig(level=logging.INFO)  # 设置日志级别


def list_bbc_search_page_urls():
    # covid; opening; closing; village; fair, china
    selenium_query.list_bbc_search_page_urls('olympic tokyo china')


def separate_urls_file_to_sub_files():
    bbc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_BBC)
    list_dir = os.path.join(bbc_dir, constant.DIR_LIST)
    urls_filename = 'urls.txt'
    lines = file.read_file_lines(list_dir, urls_filename)
    sub_file_count = 10
    sub_lines = []
    iterator = 0
    filename_index = 1
    for line in lines:
        iterator += 1
        if line.endswith('\n'):
            line = line.replace('\n', '')
        sub_lines.append(line)
        if iterator == sub_file_count:
            file.write_lines_to_file(list_dir, str(filename_index) + '.txt', sub_lines, 'w')
            filename_index += 1
            iterator = 0
            sub_lines.clear()


def bbc_news_detail_to_db():
    bbc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_BBC)
    list_dir = os.path.join(bbc_dir, constant.DIR_LIST)
    processed_dir = os.path.join(bbc_dir, constant.DIR_PROCESSED)
    list_files = file.list_dirs(list_dir)
    driver = None
    for file_name in list_files:
        if '.txt' not in file_name:
            continue
        now_time = datetime.datetime.now()
        logger.info("filename:" + file_name + "###start###" + now_time.strftime('%Y-%m-%d %H:%M:%S'))
        try:
            driver = process_bbc_news_detail_urls_file(driver, list_dir, file_name)
            # move the file to processed
            file.mv_file(list_dir, file_name, processed_dir, file_name)
        except:
            logger.info("filename:" + file_name + "*******FAIL*******")
        now_time = datetime.datetime.now()
        logger.info("filename:" + file_name + "###end###" + now_time.strftime('%Y-%m-%d %H:%M:%S'))
        # break


def process_bbc_news_detail_urls_file(driver, file_dir, file_name):
    lines = file.read_file_lines(file_dir, file_name)
    failed_urls = []
    data_list = []
    db_obj = db.DB()
    for url in lines:
        try:
            logger.info('url: ' + url)
            # open the detail page by url
            driver = selenium_query.open_page(driver, url, sleep_time=2)
            # date: 2021-08-09T03:28:15.000Z
            # date: 2021-08-08T18:00:00+01:00
            page_result = selenium_query.get_bbc_detail_content(driver, url)
            status = page_result['status']
            # get the content from the page
            if status == 'fail':
                failed_urls.append(url)
            else:
                data = page_result['data']
                date = utils.date_string_to_datetime(data['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
                data['date'] = date
                data_list.append(data)
        except:
            logger.info("url:" + url + "*******FAIL*******")
            failed_urls.append(url)
    if len(failed_urls) > 0:
        abc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_BBC)
        file.write_lines_to_file(abc_dir, constant.FILE_FAIL, failed_urls, 'a')
    db_obj.insert_news(data_list)
    db_obj.close_conn()
    return driver

# list_bbc_search_page_urls()
# separate_urls_file_to_sub_files()
# bbc_news_detail_to_db()