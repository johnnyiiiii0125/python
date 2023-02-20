import selenium_query
import os
import file
import db
from common import constant, utils
import logging
import datetime


logger = logging.getLogger('abc')
logging.basicConfig(level=logging.INFO)  # 设置日志级别


def list_abc_search_page_urls():
    selenium_query.list_abc_search_page_urls('olympic tokyo', page=1)


def abc_news_detail_to_db():
    abc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_ABC)
    list_dir = os.path.join(abc_dir, constant.DIR_LIST)
    processed_dir = os.path.join(abc_dir, constant.DIR_PROCESSED)
    list_files = file.list_dirs(list_dir)
    driver = None
    for file_name in list_files:
        now_time = datetime.datetime.now()
        logger.info("filename:" + file_name + "###start###" + now_time.strftime('%Y-%m-%d %H:%M:%S'))
        if '.txt' not in file_name:
            continue
        try:
            driver = process_abc_news_detail_urls_file(driver, list_dir, file_name)
            # move the file to processed
            file.mv_file(list_dir, file_name, processed_dir, file_name)
        except:
            logger.info("filename:" + file_name + "*******FAIL*******")
        now_time = datetime.datetime.now()
        logger.info("filename:" + file_name + "###end###" + now_time.strftime('%Y-%m-%d %H:%M:%S'))
        # break


def process_abc_news_detail_urls_file(driver, file_dir, file_name):
    lines = file.read_file_lines(file_dir, file_name)
    failed_urls = []
    data_list = []
    db_obj = db.DB()
    for url in lines:
        # print('url: ' + url)
        # open the detail page by url
        driver = selenium_query.open_page(driver, url, sleep_time=2)
        # date: 2021-07-29T08:05+1000
        # date: 2021-07-27T03:42:35.000Z
        page_result = selenium_query.get_abc_detail_content(driver, url)
        status = page_result['status']
        # get the content from the page
        if status == 'fail':
            failed_urls.append(url)
        else:
            data = page_result['data']
            date = utils.date_string_to_datetime(data['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            data['date'] = date
            data_list.append(data)
    if len(failed_urls) > 0:
        abc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_ABC)
        file.write_lines_to_file(abc_dir, constant.FILE_FAIL, failed_urls, 'a')
    db_obj.insert_news(data_list)
    db_obj.close_conn()
    return driver


def download_abc_news_detail():
    """not used"""
    abc_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_ABC)
    list_dir = os.path.join(abc_dir, constant.DIR_LIST)
    detail_dir = os.path.join(abc_dir, constant.DIR_DETAIL)
    list_files = file.list_dirs(list_dir)
    detail_files = file.list_dirs(detail_dir)
    driver = None
    for file_name in list_files:
        dir_name = file_name.replace('.txt', '')
        if dir_name in detail_files:
            continue
        logger.info("filename:" + file_name)
        if '.txt' not in file_name:
            continue
        lines = file.read_file_lines(list_dir, file_name)
        for url in lines:
            driver = selenium_query.open_page(driver, url)
            file.write_to_file(os.path.join(detail_dir, dir_name),
                               str(hash(url)) + '.html',
                               driver.page_source.encode("gbk", "ignore"), 'wb')


list_abc_search_page_urls()
abc_news_detail_to_db()