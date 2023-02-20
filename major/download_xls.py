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


url_prefix = 'https://www.zjzs.net/moban/index/'
url = url_prefix + '2c9081f061d15b160161d1653f2e000e_tree.html###'
response = get_page_response(url)
soup = get_soup(response.text)
# print(soup)
print(soup.iframe['src'])
