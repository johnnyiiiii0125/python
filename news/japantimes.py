import time

from selenium import webdriver
from common import selenium_custom
from common import constant
from common import utils
import db
import selenium_query
import logutil

class JapanTimes:
    def __init__(self):
        self.search_url = "https://www.japantimes.co.jp/search?query=asian+games&section=all&content=all&qsort=newest&pgno="
        self.driver = selenium_custom.use_opened_chrome_windows()
        self.source = constant.SOURCE_THE_JAPAN_TIMES
        self.logger = logutil.Logger("japantimes")

    def get_search_page(self, page=1):
        self.driver.get(self.search_url + str(page))
        self.get_search_page_urls(page=page)

    def get_search_page_urls(self, page=1):
        time.sleep(3)
        print('____________________Page:' + str(page) + "-----start")
        search_div = self.driver.find_element_by_xpath('//div[contains(@class, "jt-search-widget")]')
        search_results = selenium_custom.safe_find_elm(search_div, './/div[@class="search-results"]')
        if search_results is None:
            print("search-results not exists.")
            return
        articles = selenium_custom.safe_find_elms(search_results, './/div[@class="article"]')
        if articles is None or len(articles) == 0:
            print("No results!")
            return
        i = 0
        data_list = []
        db_obj = db.DB()
        for article in articles:
            i += 1
            print("article: " + str(i))
            data = {'source': self.source}
            url = ''
            article_title_h2 = selenium_custom.safe_find_elm(article, './/h2[@class="article-title"]')
            if article_title_h2 is not None:
                a_tag = article_title_h2.find_element_by_tag_name('a')
                url = a_tag.get_attribute('href')
                print(url)
            data['url'] = url
            data_list.append(data)
        db_obj.insert_news_urls(data_list)
        db_obj.close_conn()
        print('____________________Page:' + str(page) + "-----end")
        #下一页
        self.get_search_page(page+1)

    def get_news_detail(self):
        page = 1
        urls = []
        while True:
            urls = self.get_news_urls_from_db(page)
            if len(urls) == 0:
                break
            print("#######Page: " + str(page) + "######")
            self.logger.log_now("Start get " + str(constant.LIMIT) + " news detail")
            data_list = []
            failed_news_urls = []
            for url in urls:
                print(url)
                selenium_query.open_page(self.driver, url, sleep_time=2)
                page_data = self.get_detail_content(url)
                if page_data is None:
                    failed_news_urls.append({'source': self.source, 'url': url})
                else:
                    data_list.append(page_data)
            self.logger.log_now("Finish get " + str(constant.LIMIT) + " news detail")
            self.logger.log_now("Start insert news detail")
            db_obj = db.DB()
            if len(data_list) > 0:
                db_obj.insert_news(data_list)
            if len(failed_news_urls) > 0:
                db_obj.insert_failed_news_urls(failed_news_urls)
            db_obj.close_conn()
            self.logger.log_now("Finish insert news detail")
            page = page + 1

    def get_detail_content(self, url):
        # (source, url, title, content, date)
        page_data = {'source': self.source, 'url': url}
        title_article_h1 = self.driver.find_element_by_xpath('//h1[@class="title-article"]')
        if title_article_h1 is None:
            print("title_article_h1 is None")
            return None
        page_data['title'] = title_article_h1.text
        jt_article_details = self.driver.find_element_by_xpath('//div[contains(@class, "jt-article-details")]')
        if jt_article_details is None:
            print("jt_article_details is None")
            return None
        article_body_div = selenium_custom.safe_find_elm(jt_article_details, './/div[@class="article-body"]')
        if article_body_div is None:
            print("article_body_div is None")
            return None
        page_data['content'] = article_body_div.text
        publish_date_div = selenium_custom.safe_find_elm(jt_article_details, './/div[@class="publish-date"]')
        if publish_date_div is None:
            print("publish_date_div is None")
            return None
        # Dec 15, 2023
        page_data['date'] = utils.japantimes_date_string_to_datetime(publish_date_div.text)
        return page_data

    def get_news_urls_from_db(self, page):
        self.logger.log_now("Start get news urls from db")
        db_obj = db.DB()
        sql = "select * from " + db.table_news_urls + " WHERE source='" + self.source + "' LIMIT " + str(constant.LIMIT)\
              + " OFFSET " + str((page-1)*constant.LIMIT)
        result = db_obj.query_results(sql)
        data = list(result['data'])
        urls = []
        for item in data:
            urls.append(item[2])
        db_obj.close_conn()
        self.logger.log_now("Finish get news urls from db")
        return urls




if __name__ == "__main__" :
    jp = JapanTimes()
    # jp.get_search_page(page=109)
    jp.get_news_detail()



