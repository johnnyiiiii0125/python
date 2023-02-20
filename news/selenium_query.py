import time
import os
from common import constant
from common import selenium_custom
from selenium import webdriver
import urllib
import file

'''
ABC news
'''
ABC_SEARCH_URL = 'https://search-beta.abc.net.au/index.html#/?'
PAGE = 1


def get_abc_search_page_urls(driver, page=PAGE):
    time.sleep(3)
    print('Page:' + str(page))
    ul_div = driver.find_element_by_xpath('//div[@data-component="SearchHits"]')
    ul = ul_div.find_element_by_tag_name('ul')
    lis = ul.find_elements_by_tag_name('li')
    lines = []
    i = 0
    for li in lis:
        i = i + 1
        print('li:' + str(i))
        a_tag = li.find_element_by_tag_name('a')
        a_href = a_tag.get_attribute('href')
        lines.append(a_href)
    file.write_lines_to_file(os.path.join(constant.ROOT_DIR, constant.SOURCE_ABC, constant.DIR_LIST),
                             str(page) + '.txt',
                             lines)
    # 下一页
    next_btn = driver.find_element_by_xpath('//button[@data-component="Pagination__Next"]')
    if not next_btn.get_attribute('disabled'):
        next_btn.click()
        get_abc_search_page_urls(driver, page + 1)


def list_abc_search_page_urls(keyword, page=PAGE, search_url=ABC_SEARCH_URL):
    url = search_url + 'query=' + urllib.parse.quote(keyword) + '&page=' + str(page)
    driver = webdriver.Chrome()
    driver.get(url)
    get_abc_search_page_urls(driver, page)
    driver.quit()


def open_page(driver, url, sleep_time=3):
    if driver is None:
        driver = selenium_custom.use_opened_chrome()
    driver.get(url)
    time.sleep(sleep_time)
    return driver


def get_abc_detail_content(driver, url=None):
    result = {}
    data = {'url': url, 'source': constant.SOURCE_ABC}
    result['data'] = data
    result['status'] = 'success'
    # audio
    audio_div = selenium_custom.safe_find_elm(driver, '//div[@id="audioPlayerWithDownload4"]')
    if audio_div is not None:
        result['page_type'] = 'audio'
        h1_name = selenium_custom.safe_find_elm(driver, '//h1[@itemprop="name"]')
        if h1_name is not None:
            data['title'] = h1_name.text
        else:
            result['status'] = 'fail'
        text_div = selenium_custom.safe_find_elm(driver, '//div[@id="comp-rich-text8"]')
        if text_div is not None:
            data['content'] = text_div.text
        else:
            data['content'] = ''
        date_div = selenium_custom.safe_find_elm(driver, '//div[@itemprop="datePublished"]')
        if date_div is not None:
            date = date_div.get_attribute('content')
            data['date'] = date
        else:
            data['date'] = ''
            result['status'] = 'fail'
        return result
    # video
    video_div = selenium_custom.safe_find_elm(driver, '//div[@id="video-player4"]')
    if video_div is not None:
        result['page_type'] = 'video'
        h1_name = selenium_custom.safe_find_elm(driver, '//h1[@itemprop="name"]')
        if h1_name is not None:
            data['title'] = h1_name.text
        else:
            result['status'] = 'fail'
        transcript_div = selenium_custom.safe_find_elm(driver, '//div[@class="view-transcript"]')
        if transcript_div is not None:
            btn_div = selenium_custom.safe_find_elm(transcript_div, './/div[@role="button"]')
            if btn_div is not None:
                btn_div.click()
                time.sleep(0.5)
            text_div = selenium_custom.safe_find_elm(transcript_div, './/div[@id="comp-rich-text8"]')
            data['content'] = text_div.text
        else:
            data['content'] = ''
        date_div = selenium_custom.safe_find_elm(driver, '//div[@itemprop="datePublished"]')
        if date_div is not None:
            date = date_div.get_attribute('content')
            data['date'] = date
        else:
            data['date'] = ''
            result['status'] = 'fail'
        return result
    audio_div = selenium_custom.safe_find_elm(driver, '//div[@data-component="Player"]')
    if audio_div is not None:
        result['page_type'] = 'audio'
    # text
    h1_name = selenium_custom.safe_find_elm(driver, '//h1[@data-component="Heading"]')
    if h1_name is not None:
        data['title'] = h1_name.text
    else:
        result['status'] = 'fail'
    if audio_div is None:
        body_div = selenium_custom.safe_find_elm(driver, '//div[@id="body"]')
        if body_div is not None:
            container_div = selenium_custom.safe_find_elm(body_div, '//div[@data-component="LayoutContainer"]')
            content_div = selenium_custom.safe_find_elms_tag(container_div, 'div')[0]
            data['content'] = content_div.text
        else:
            data['content'] = ''
    else:
        container_div = selenium_custom.safe_find_elm(driver, '//div[@data-component="LayoutContainer"]')
        content_div = selenium_custom.safe_find_elms_tag(container_div, 'div')[0]
        data['content'] = content_div.text
    date_div = selenium_custom.safe_find_elm(driver, '//div[@data-component="Dateline"]')
    if date_div is not None:
        time_elm = selenium_custom.safe_find_elm(date_div, '//time[@data-component="Timestamp"]')
        if time_elm is not None:
            date = time_elm.get_attribute('datetime')
            data['date'] = date
        else:
            data['date'] = ''
            result['status'] = 'fail'
    else:
        data['date'] = ''
        result['status'] = 'fail'
    return result


'''
BBC news
'''
BBC_SEARCH_URL = 'https://www.bbc.co.uk/search?'


def list_bbc_search_page_urls(keyword, page=PAGE, search_url=BBC_SEARCH_URL):
    url = search_url + 'q=' + urllib.parse.quote(keyword) + '&page=' + str(page)
    driver = selenium_custom.use_opened_chrome()
    driver.get(url)
    get_bbc_search_page_urls(driver, page)
    driver.quit()


def get_bbc_search_page_urls(driver, page=PAGE):
    time.sleep(3)
    print('Page:' + str(page))
    main_elm = selenium_custom.safe_find_elm(driver, '//main[@id="main-content"]')
    ul = selenium_custom.safe_find_elms_tag(main_elm, 'ul')[0]
    lis = ul.find_elements_by_xpath('li')
    print('items count: ' + str(len(lis)))
    lines = []
    i = 0
    for li in lis:
        i = i + 1
        print('li:' + str(i))
        a_tag = li.find_element_by_tag_name('a')
        a_href = a_tag.get_attribute('href')
        lines.append(a_href)
    # check the urls are already in the file
    file_dir = os.path.join(constant.ROOT_DIR, constant.SOURCE_BBC, constant.DIR_LIST, )
    processed_lines = file.remove_redundant_urls_in_file(lines, os.path.join(file_dir, 'urls.txt'))
    file.write_lines_to_file(file_dir, 'urls.txt',
                             processed_lines, 'a')
    # 下一页
    nav_elm = selenium_custom.safe_find_elm(main_elm, './/nav[@aria-label="Page"]')
    if nav_elm is not None:
        arrow_btns = selenium_custom.safe_find_elms(nav_elm,
                                                  './/div[contains(@class, "ssrcss-a11ok4-ArrowPageButtonContainer")]')
        if arrow_btns is not None and len(arrow_btns) == 2:
            next_btn = arrow_btns[1]
            next_a = selenium_custom.safe_find_elm(next_btn, './/a')
            if next_a is not None:
                next_a.click()
                get_bbc_search_page_urls(driver, page + 1)


'''
return:
{
    "status": "success",
    "data": {
        "url": "",
        "source": "bbc",
        "title": "",
        "content": "",
        "date": "",
    }
}
'''


def get_bbc_detail_content(driver, url=None):
    result = {}
    data = {'url': url, 'source': constant.SOURCE_BBC}
    result['data'] = data
    result['status'] = 'success'
    video_article = selenium_custom.safe_find_elm(driver, './/article[contains(@class, "StyledMediaItem")]')
    # video
    if video_article is not None:
        title_h1 = selenium_custom.safe_find_elm(video_article, './/h1[@id="main-heading"]')
        if title_h1 is not None:
            data['title'] = title_h1.text
        else:
            result['status'] = "fail"
        content_div = selenium_custom.safe_find_elm(video_article, './/div[contains(@class, "StyledSummary")]')
        if content_div is not None:
            data['content'] = content_div.text
        else:
            data['content'] = ''
        time_elm = selenium_custom.safe_find_elm(video_article, './/time[@data-testid="timestamp"]')
        if time_elm is not None:
            data['date'] = time_elm.get_attribute('datetime')
        else:
            result['status'] = "fail"
        return result
    # programmes
    programmes_div = selenium_custom.safe_find_elm(driver, './/div[@id="programmes-content"]')
    if programmes_div is not None:
        programme_detail_div = selenium_custom.safe_find_elm(driver, './/div[@data-map-column="playout-details"]')
        if programme_detail_div is not None:
            title_h1 = selenium_custom.safe_find_elm(programme_detail_div, './/h1[@class="no-margin"]')
            if title_h1 is None:
                title_div = selenium_custom.safe_find_elm(driver, './/div[@class="br-masthead__title"]')
                data['title'] = title_div.text
            else:
                data['title'] = title_h1.text
        else:
            result['status'] = "fail"
        content_div = selenium_custom.safe_find_elm(driver, './/div[@class="grid-wrapper"]')
        if content_div is not None:
            toggle_btn = selenium_custom.safe_find_elm(content_div, './/label[contains(@class, "synopsis-toggle__button")]')
            if toggle_btn is not None:
                toggle_btn.click()
            data['content'] = content_div.text
        else:
            data['content'] = ''
        map_inner_div = selenium_custom.safe_find_elm(driver, './/div[@class="map__inner"]')
        if map_inner_div is not None:
            date_div = selenium_custom.safe_find_elm(map_inner_div, './/div[contains(@class, "broadcast-event__time")]')
            if date_div is not None:
                data['date'] = date_div.get_attribute('content')
            else:
                result['status'] = "fail"
        else:
            result['status'] = "fail"
        return result
    # others
    article_elm = selenium_custom.safe_find_elm(driver, './/article')
    if article_elm is not None:
        header = selenium_custom.safe_find_elm(article_elm, 'header')
        if header is not None:
            title_h1 = selenium_custom.safe_find_elm(header, 'h1')
            data['title'] = title_h1.text
            time_elm = selenium_custom.safe_find_elm(header, './/time')
            if time_elm is not None:
                data['date'] = time_elm.get_attribute('datetime')
            else:
                result['status'] = "fail"
        else:
            result['status'] = "fail"
        text_blocks = selenium_custom.safe_find_elms(article_elm, 'div[@data-component="text-block"]')
        content = ''
        if text_blocks is None or len(text_blocks) == 0:
            content_div = selenium_custom.safe_find_elm(article_elm, 'div')
            content = content_div.text
        else:
            for text_block in text_blocks:
                content += text_block.text
        data['content'] = content
    else:
        result['status'] = "fail"
    return result
