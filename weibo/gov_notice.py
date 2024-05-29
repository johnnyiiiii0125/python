import time
import os
from common import selenium_custom
import json
import db
import pandas as pd
from common import utils
import constant


class SeleniumSearch:
    def __init__(self, driver_needed=True):
        if driver_needed:
            self.driver = selenium_custom.use_opened_chrome_windows()
        self.url = ''
        self.dbobj = db.DB()

    def init_url(self, userid, keyword, page=1):
        # self.url = 'https://weibo.com/u/' + user_list[username] + \
        # '?key_word=' + keyword + '&is_ori=1&is_text=1&is_pic=1&is_video=1&is_music=1' + \
        #                         '&start_time=1577721600&end_time=1676563200'

        self.url = 'https://weibo.com/ajax/statuses/searchProfile?uid=' + str(userid) + '&page=' + str(page)\
                   + '&q=' + keyword + '&hasori=1&hastext=1&haspic=1&hasvideo=1&hasmusic=1' \
                                  + '&starttime=1577721600&endtime=1676563200'

    def process_list_data(self, userid, keyword, page):
        time.sleep(3)
        self.init_url(userid, keyword, page)
        self.driver.get(self.url)
        pre_elm = self.driver.find_element_by_tag_name('pre')
        json_text = pre_elm.text
        data = json.loads(json_text)
        datalist = data['data']['list']
        if len(datalist) == 0:
            return
        print(keyword + ', ' + str(userid) + ',' + str(page))
        db_list = []
        for item in datalist:
            db_item = {}
            mblogid = item['mblogid']
            mid = item['mid']
            user = item['user']
            db_item['weiboid'] = mblogid
            db_item['mid'] = mid
            db_item['userid'] = user['id']
            db_item['username'] = user['screen_name']
            db_item['keyword'] = keyword
            db_list.append(db_item)
        self.dbobj.get_conn()
        self.dbobj.insert_weibolist(db_list)
        self.dbobj.close_conn()
        self.process_list_data(userid, keyword, page + 1)

    def do_search(self, keyword, page):
        df = pd.read_excel(os.path.join(constant.ROOT_DIR, constant.GOV_WEIBO_FILENAME))
        for idx, row in df.iterrows():
            userid = row['userid']
            hasfetched = row['hasfetched']
            print("微博用户：" + row['name'])
            if pd.isna(userid) or hasfetched == 1:
                continue
            self.process_list_data(userid, keyword, page)
            df.loc[idx, 'hasfetched'] = 1
            df.to_excel(os.path.join(constant.ROOT_DIR, constant.GOV_WEIBO_FILENAME), index=False)

    def do_search_mid(self, keyword, page):
        df = pd.read_excel(os.path.join(constant.ROOT_DIR, constant.GOV_WEIBO_FILENAME))
        for idx, row in df.iterrows():
            # userid = row['userid']
            # hasfetched = row['hasfetched']
            # print("微博用户：" + row['name'])
            # if pd.isna(userid) or hasfetched == 0:
            #     continue
            userid = '5279514376'
            self.process_list_mid(userid, keyword, page)

    def process_list_mid(self, userid, keyword, page):
        time.sleep(3)
        self.init_url(userid, keyword, page)
        self.driver.get(self.url)
        pre_elm = self.driver.find_element_by_tag_name('pre')
        json_text = pre_elm.text
        data = json.loads(json_text)
        datalist = data['data']['list']
        if len(datalist) == 0:
            return
        print(keyword + ', ' + str(userid) + ',' + str(page))
        db_list = []
        for item in datalist:
            db_item = {}
            mblogid = item['mblogid']
            mid = item['mid']
            db_item['mid'] = mid
            db_item['weiboid'] = mblogid
            db_list.append(db_item)
        self.dbobj.get_conn()
        self.dbobj.update_mid(db_list)
        self.dbobj.close_conn()
        self.process_list_mid(userid, keyword, page + 1)

    def get_detail(self, page):
        while True:
            print('page----------' + str(page))
            # get 20 from list
            data_list = self.dbobj.query_weibo_list(page=page)
            if len(data_list) == 0:
                return
            ids = []
            detail_list = []
            for list_item in data_list:
                ids.append(list_item[0])
                detail_list.append(self.process_detail_page(list_item))
            # insert into detail
            self.dbobj.insert_detail(detail_list)
            # mark the 20 as fetched
            self.dbobj.mark_list_as_processed(ids)
            self.dbobj.close_conn()
            page += 1

    def process_detail_page(self, list_item):
        weiboid = list_item[1]
        userid = list_item[2]
        username = list_item[3]
        keyword = list_item[4]
        mid = list_item[5]
        print('weibo id-------' + weiboid + '----------username------' + username)
        url = 'https://weibo.com/' + userid + '/' + weiboid
        self.driver.get(url)
        time.sleep(2)
        detail_item = {'weiboid': weiboid, 'userid': userid,
                       'username': username, 'keyword': keyword, 'mid': str(mid)}
        article = self.driver.find_element_by_xpath('.//article[contains(@class, "Detail_feed_3iffy")]')
        feed_body = article.find_element_by_xpath('.//div[@class="Feed_body_3R0rO"]')
        # header
        header = feed_body.find_element_by_xpath('.//header[@class="woo-box-flex"]')
        a_date = header.find_element_by_xpath('.//a[@class="head-info_time_6sFQg"]')
        detail_item['url'] = a_date.get_attribute('href')
        date_txt = utils.weibo_time_string_to_datetime(a_date.text)
        detail_item['time'] = date_txt
        # content
        div_content_wrapper = feed_body.find_element_by_xpath('.//div[@class="wbpro-feed-content"]')
        div_content = div_content_wrapper.find_element_by_xpath('.//div[@class="detail_wbtext_4CRf9"]')
        detail_item['content'] = div_content.text
        # footer
        footer = article.find_element_by_tag_name('footer')
        footer_btns_wrapper = footer.find_element_by_xpath('.//div[contains(@class, "toolbar_main_3Mxwo")]')
        footer_btns = footer_btns_wrapper.find_elements_by_xpath('.//div[contains(@class, "toolbar_cursor_34j5V")]')
        detail_item['forward_count'] = self.get_count(footer_btns[0].text)
        detail_item['reply_count'] = self.get_count(footer_btns[1].text)
        detail_item['like_count'] = self.get_count(footer_btns[2].text)
        return detail_item

    def get_count(self, count_txt):
        count_txt = count_txt.strip()
        if '转发' == count_txt or '评论' == count_txt or '赞' == count_txt:
            return 0
        if '万' in count_txt:
            count_txt = count_txt.replace('万', '')
            return float(count_txt) * 10000
        return int(count_txt)


ss = SeleniumSearch()
# ss.do_search('疫情', page=1)
ss.get_detail(1)
# ss.do_search_mid('疫情', page=1)