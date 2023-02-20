


import  requests

for i in range(1, 5):
    for j in range(1, 5):
        url = 'http://challenge-2e682632728d747e.sandbox.ctfhub.com:10800/flag_in_here/' \
              + str(i) + "/" + str(j)
        resp = requests.get(url)
        content = resp.text
        content = content.replace("flag_in_here", '')
        if "flag" in content:
            print(url)