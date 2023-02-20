import time
from selenium import webdriver
from common import selenium_custom
import urllib
import datetime
import db

WEIBO_LOGIN_URL = 'https://weibo.com/login.php'
WEIBO_SEARCH_URL = 'https://s.weibo.com/weibo/?'
PAGE = 1
SEARCH_KEYWORD = '工匠精神'

STOP_YEAR = 2021
STOP_MONTH = 6
STOP_DAY = 1


def do_login():
    driver = webdriver.Chrome()
    driver.get(WEIBO_LOGIN_URL)
    time.sleep(10)
    loginname = driver.find_element_by_id('loginname')
    password = driver.find_element_by_name('password')
    login_btn = driver.find_element_by_xpath('//a[@action-type="btn_submit"]')
    loginname.send_keys('')
    password.send_keys('')
    login_btn.click()
    time.sleep(10)
    return driver





def get_forward_reply_count(li):
    txt = li.text
    if '转发' == txt or '评论' == txt:
        return 0
    else:
        return int(txt.strip())


def get_like_count(li):
    span = li.find_element_by_xpath('.//span[@class="woo-like-count"]')
    txt = span.text
    if '赞' == txt:
        return 0
    else:
        return int(txt.strip())


def get_poster(result, card_feed):
    avator = card_feed.find_element_by_class_name('avator')  # 头像
    avator_a = avator.find_element_by_tag_name('a')
    result['poster_home'] = avator_a.get_attribute('href')
    result['poster_id'] = get_poster_id(result['poster_home'])
    verify_span = selenium_custom.safe_find_elm(avator_a, './span')
    if verify_span is not None:
        result['poster_verify'] = verify_span.get_attribute('title')
    else:
        result['poster_verify'] = ''
    content = card_feed.find_element_by_class_name('content')
    info = content.find_element_by_class_name('info')  # 用户信息
    result['poster_name'] = info.find_element_by_xpath('.//a[@nick-name]').get_attribute('nick-name')


def get_poster_id(poster_home):
    subs = poster_home.split('?')
    if len(subs) == 2:
        subss = subs[0].split('/')
        return subss[len(subss) - 1]
    return ''


def get_original(result, content):
    post_original_url = ''
    card_comment = selenium_custom.safe_find_elm(content, './/div[@class="card-comment"]')  # 转发内容
    if card_comment is not None:
        card_comment_p = card_comment.find_element_by_xpath('.//p[@class="from"]')
        if card_comment_p is not None:
            post_original_url = card_comment_p.find_elements_by_tag_name('a')[0].get_attribute('href')
    result['post_original_url'] = post_original_url


def next_day(t):
    nums = t.split('-')
    year = int(nums[0])
    month = int(nums[1])
    day = int(nums[2])
    if day == get_month_max_day(year, month):
        month += 1
        day = 1
        if month > 12:
            year += 1
            month = 1
    else:
        day += 1
    now_time = datetime.datetime.now()
    n_year = now_time.year
    n_month = now_time.month
    n_day = now_time.day
    if year >= STOP_YEAR and month >= STOP_MONTH and day >= STOP_DAY:
        return None
    return str(year) + '-' + get_m_d_str(month) + '-' + get_m_d_str(day) + '-' + nums[3]


def get_year_month_day(t):
    nums = t.split('-')
    year = int(nums[0])
    month = int(nums[1])
    day = int(nums[2])
    return {'year': year, 'month': month, 'day': day}


def get_m_d_str(v):
    if v < 10:
        return '0' + str(v)
    return str(v)


def get_month_max_day(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    if month == 2:
        if is_run_nian(year):
            return 29
        else:
            return 28


def is_run_nian(year):
    if year % 4 == 0 and year % 100 != 0:
        return True
    if year % 400 == 0:
        return True
    return False


def do_search(driver=None, keyword=SEARCH_KEYWORD, page=PAGE, search_url=WEIBO_SEARCH_URL, from_time='2016-03-06-0', end_time='2016-03-06-23', db_obj=None):
    url = search_url + 'q=' + urllib.parse.quote(keyword) + '&typeall=1&suball=1' + '&timescope=custom:' + from_time + ':' + end_time + '&Refer=q&page=' + str(page)
    if driver is None:
        driver = webdriver.Chrome()
    driver.get(url)
    return process_items(driver, keyword, from_time, end_time, db_obj, True, page)


def next_time_range(driver, keyword, from_time, end_time, db_obj):
    # 没有其他结果
    now_time = datetime.datetime.now()
    print('This time range ended...' + from_time + ':' + end_time + '####Now: '
          + now_time.strftime('%Y-%m-%d %H:%M:%S'))
    f_time = next_day(from_time)
    e_time = next_day(end_time)
    if f_time is not None and e_time is not None:
        return do_search(driver, keyword=keyword, page=PAGE, search_url=WEIBO_SEARCH_URL, from_time=f_time,
                  end_time=e_time, db_obj=db_obj)
    return None


def print_start_time(m_page, from_time, end_time):
    if m_page is not None:
        p_lis = selenium_custom.safe_find_elms_tag(m_page, 'li')
        if p_lis is not None:
            now_time = datetime.datetime.now()
            global max_page
            max_page = len(p_lis)
            print('This time range start...' + from_time + ':' + end_time + '####Now: '
                  + now_time.strftime('%Y-%m-%d %H:%M:%S') + '####共' + str(len(p_lis)) + '页')


def process_items(driver, keyword, from_time, end_time, db_obj, first_in, cur_page):
    time.sleep(5)
    # 确认是否已到最后一页
    card_no_result = selenium_custom.safe_find_elm(driver, '//img[@class="no-result"]')
    if card_no_result is not None:
        next_time_range(driver, keyword, from_time, end_time, db_obj)
        return None
    card_no_result = selenium_custom.safe_find_elm(driver, '//div[@class="card card-no-result s-pt20b40"]')
    if card_no_result is not None:
        next_time_range(driver, keyword, from_time, end_time, db_obj)
        return None
    if cur_page > max_page:# 超过最大页？？？需确认
        next_time_range(driver, keyword, from_time, end_time, db_obj)
        return None
    # 获取共多少页
    m_page = selenium_custom.safe_find_elm(driver, '//div[@class="m-page"]')
    if first_in:
        print_start_time(m_page, from_time, end_time)
    item_list = driver.find_elements_by_xpath('//div[@action-type="feed_list_item"]')
    results = []
    year_month_day = get_year_month_day(from_time)
    for item_div in item_list:
        result = {}
        result['keyword'] = keyword
        result['post_year'] = year_month_day['year']
        result['post_month'] = year_month_day['month']
        result['post_day'] = year_month_day['day']
        card = item_div.find_element_by_class_name('card')
        card_feed = card.find_element_by_class_name('card-feed')
        content = card_feed.find_element_by_class_name('content')
        get_poster(result, card_feed)
        p_from = content.find_element_by_class_name('from')  # 发布时间  以及设备
        p_from_a = p_from.find_elements_by_tag_name('a')
        p_from_time = p_from_a[0]
        result['post_time'] = p_from_time.text
        result['post_url'] = p_from_time.get_attribute('href')
        if len(p_from_a) > 1:
            result['post_device'] = p_from_a[1].text
        else:
            result['post_device'] = ''
        feed_list_content = content.find_element_by_xpath('p[@node-type="feed_list_content"]')  # 微博内容
        result['post_content'] = feed_list_content.text
        feed_list_media_prev = selenium_custom.safe_find_elm(content, 'div[@node-type="feed_list_media_prev"]')
        img_urls = []
        if feed_list_media_prev is not None:
            imgs = feed_list_media_prev.find_elements_by_tag_name('img')
            if imgs is not None:
                for img in imgs:
                    img_urls.append(img.get_attribute('src'))
        # TODO: videos
        result['post_media'] = {'image_urls': img_urls}
        # actions
        card_act = card.find_element_by_class_name('card-act')
        act_lis = card_act.find_elements_by_tag_name('li')
        forward_count = 0
        like_count = 0
        reply_count = 0
        if act_lis is not None:
            if 3 == len(act_lis):
                forward_count = get_forward_reply_count(act_lis[0])
                reply_count = get_forward_reply_count(act_lis[1])
                like_count = get_like_count(act_lis[2])
        result['post_forward_count'] = forward_count
        result['post_like_count'] = like_count
        result['post_reply_count'] = reply_count
        # forwarded
        get_original(result, content)
        results.append(result)
    db_obj.insert_results(results)
    if m_page is not None:
        next_btn = selenium_custom.safe_find_elm(m_page, './/a[@class="next"]')
        if next_btn is not None:
            next_btn.click()
            return process_items(driver, keyword, from_time, end_time, db_obj, False, cur_page+1)
        else:
            return next_time_range(driver, keyword, from_time, end_time, db_obj)
    else:
        return next_time_range(driver, keyword, from_time, end_time, db_obj)


max_page = 1
db_obj = db.DB()
db_obj.get_conn()
driver = selenium_custom.use_opened_chrome()
do_search(driver, keyword=SEARCH_KEYWORD, page=1, from_time='2021-05-27-0', end_time='2021-05-27-23', db_obj=db_obj)
db_obj.close_conn()
