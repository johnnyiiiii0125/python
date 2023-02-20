import os
import urllib
import requests  #导入requests包
from bs4 import BeautifulSoup
import random
import logging
from itertools import cycle
import file
import json
from common import constant

logger = logging.getLogger('news_query')
logging.basicConfig(level=logging.INFO)  # 设置日志级别

HEADERS_LIST = [
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; x64; fr; rv:1.9.2.13) Gecko/20101203 Firebird/3.6.13',
    'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
    'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16',
    'Mozilla/5.0 (Windows NT 5.2; RW; rv:7.0a1) Gecko/20091211 SeaMonkey/9.23a1pre'
]
HEADER = {'User-Agent': random.choice(HEADERS_LIST), 'X-Requested-With': 'XMLHttpRequest'}

PROXY_URL = 'https://free-proxy-list.net/'


def get_proxies():
    response = requests.get(PROXY_URL)
    soup = BeautifulSoup(response.text, 'lxml')
    #table = soup.find('table',id='proxylisttable')
    table = soup.find('table')
    list_tr = table.find_all('tr')
    list_td = [elem.find_all('td') for elem in list_tr]
    list_td = list(filter(None, list_td))
    list_ip = [elem[0].text for elem in list_td]
    list_ports = [elem[1].text for elem in list_td]
    list_proxies = [':'.join(elem) for elem in list(zip(list_ip, list_ports))]
    return list_proxies


proxies = get_proxies()
proxy_pool = cycle(proxies)


'''
    CNN related parameters
    sample url : https://www.cnn.com/search?size=10&q=olympic%20tokyo&page=1&from=0&sort=relevance;
    https://search.api.cnn.io/content?size=10&sort=relevance&page=1&from=0&q=olympic tokyo
'''
CNN_SEARCH_URL = 'https://www.cnn.com/search?'
CNN_API_SEARCH_URL = 'https://search.api.cnn.io/content?'
PAGE = 1
CNN_FROM = 0
CNN_SORT = 'relevance'
CNN_Q = ''


def get_response(url, params, use_proxy):
    try:
        if use_proxy:
            proxy = next(proxy_pool)
            logger.info('Using proxy {}'.format(proxy))
            response = requests.get(url, params=params, headers=HEADER, proxies={"http": proxy})
        else:
            response = requests.get(url, params=params, headers=HEADER)
        return response
    except requests.exceptions.HTTPError as e:
        logger.exception('HTTPError {} while requesting "{}"'.format(
            e, url))
    except requests.exceptions.ConnectionError as e:
        logger.exception('ConnectionError {} while requesting "{}"'.format(
            e, url))
    except requests.exceptions.Timeout as e:
        logger.exception('TimeOut {} while requesting "{}"'.format(
            e, url))


def list_cnn_news_html(keyword, page=PAGE, cnn_from=CNN_FROM, size=constant.SIZE, sort=CNN_SORT,
                  search_url=CNN_SEARCH_URL, use_proxy=True):
    url = search_url + 'size=' + str(size) + '&sort=' + sort + '&page=' + str(page) \
          + '&from=' + str(cnn_from) + '&q=' + urllib.parse.quote(keyword)
    logger.info(url)
    response = get_response(url, {}, use_proxy)
    html = response.text or ''
    file.write_to_file(os.path.join(constant.ROOT_DIR, constant.SOURCE_CNN, constant.DIR_LIST), str(page) + '.html', html)
    #not used


def api_search_cnn_news(keyword, page=PAGE, cnn_from=CNN_FROM, size=constant.SIZE, sort=CNN_SORT,
                  search_url=CNN_API_SEARCH_URL, use_proxy=True):
    logger.info("list page:" + str(page))
    params = {
        'size': str(size),
        'sort': sort,
        'page': str(page),
        'from': str(cnn_from),
        # 'q': urllib.parse.quote(keyword)
        'q': keyword
    }
    response = get_response(search_url, params, use_proxy)
    json_text = response.text or ''
    # logger.info(html)
    data = json.loads(json_text)
    file.write_to_file(os.path.join(constant.ROOT_DIR, constant.SOURCE_CNN, constant.DIR_LIST), str(page) + '.json', json_text)
    return data

