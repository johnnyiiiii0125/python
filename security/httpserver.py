import urllib
import socketserver
from http.server import SimpleHTTPRequestHandler


class MyHandler(SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        print("got get request %s" % (self.path))
        hql = urllib.splitquery(self.path)[1]
        uri_c = str(hql)
        print('cmd===%s' % (uri_c))
        file = open(uri_c)
        s = file.read()
        print(s)
        self.wfile.write(s)
        file.close()


def start_server():
    httpd = socketserver.TCPServer(("0.0.0.0", 8090), MyHandler)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    start_server()
