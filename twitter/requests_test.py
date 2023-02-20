import requests        #导入requests包
from bs4 import    BeautifulSoup
import random

def get_page_response(url):
	user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
	headers = { 'User-Agent' : user_agent }
	response = requests.get(url, headers=headers)        #Get方式获取网页数据
	response.encoding = 'utf-8'
	return response

def get_soup(html_text):
	soup = BeautifulSoup(html_text, 'lxml')
	return soup

# url = 'http://www.baidu.com/'
# response = get_page_response(url)
# soup = get_soup(response.text)
# print(soup)

HEADERS_LIST = [
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; x64; fr; rv:1.9.2.13) Gecko/20101203 Firebird/3.6.13',
    'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
    'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16',
    'Mozilla/5.0 (Windows NT 5.2; RW; rv:7.0a1) Gecko/20091211 SeaMonkey/9.23a1pre'
]

HEADER = {'User-Agent': random.choice(HEADERS_LIST), 'X-Requested-With': 'XMLHttpRequest'}
response = requests.get('https://search-beta.abc.net.au/index.html?siteTitle=news#/?configure%5BgetRankingInfo%5D=true&configure%5BclickAnalytics%5D=true&configure%5BuserToken%5D=anonymous-7f04265c-365a-4295-93c6-8c3a7e3d3865&configure%5BhitsPerPage%5D=10&query=olympic%20tokyo&page=1', headers=HEADER)

from news import file

file.write_to_file('/Users/wangzhongyi/Library/Mobile Documents/com~apple~CloudDocs/Documents/Johnnyiiiii/教学/课程/project/news', 'test.html', response.text)

