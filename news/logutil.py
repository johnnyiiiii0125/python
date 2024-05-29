import logging
import datetime



class Logger:
    def __init__(self, title):
        self.logger = logging.getLogger(title)
        logging.basicConfig(level=logging.INFO)

    def log_now(self, message):
        now_time = datetime.datetime.now()
        self.logger.info(now_time.strftime('%Y-%m-%d %H:%M:%S') + "******" + message + "******")

    def log(self, message):
        self.logger.info("******" + message + "******")