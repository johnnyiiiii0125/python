import requests        #导入requests包
from bs4 import BeautifulSoup
import urllib.parse
import json

login_url = 'https://passport.weibo.cn/signin/login'

# define login(username, password):

def text_to_json(text):
    data = json.loads(text)
    return data

def search(keyword, page=1):
    query_url = 'https://m.weibo.cn/api/container/getIndex?containerid=100103'
    need_encoded = 'type=1&q='+keyword
    other_params = '&page_type=searchall'
    url = query_url + urllib.parse.quote(need_encoded) + other_params + '&page=' + str(page)
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)  # Get方式获取网页数据
    response.encoding = 'utf-8'
    return response


response = search('工匠精神', 1)
print(response.text)
data = text_to_json(response.text)
# print(data)


