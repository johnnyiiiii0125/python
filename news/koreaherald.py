import os.path
import time

from selenium import webdriver
from common import selenium_custom
from common import constant
from common import utils
import db
import selenium_query
import logutil
import excel
from datapreprocess import DataPreprocess
from dataprocess import DataAnalysis
import pandas as pd

'''
亚运会：2023年9月23日至10月8日举办
亚残运会：2023年10月22日至10月28日举行。
'''


class KoreaHerald:
    def __init__(self, driver_needed=True):
        self.search_url = "https://www.koreaherald.com/search/index.php?q=asian+games&sort=1&mode=list&np="
        if driver_needed:
            self.driver = selenium_custom.use_opened_chrome_windows()
        self.source = constant.SOURCE_THE_KOREA_HERALD
        self.logger = logutil.Logger("koreaherald")

    def get_search_page(self, page=1):
        self.driver.get(self.search_url + str(page))
        self.get_search_page_urls(page=page)

    def get_search_page_urls(self, page=1):
        time.sleep(3)
        print('____________________Page:' + str(page) + "-----start")
        search_result = self.driver.find_element_by_xpath('//div[@class="search_result"]')
        if search_result is None:
            print("search_result not exists.")
            return
        articles = selenium_custom.safe_find_elms(search_result, './/li[@class="sub_content_item"]')
        if articles is None or len(articles) == 0:
            print("No results!")
            return
        i = 0
        data_list = []
        db_obj = db.DB()
        for article in articles:
            i += 1
            print("article: " + str(i))
            data = {'source': self.source}
            a_tag = article.find_element_by_tag_name('a')
            url = a_tag.get_attribute('href')
            print(url)
            data['url'] = url
            data_list.append(data)
        db_obj.insert_news_urls(data_list)
        db_obj.close_conn()
        print('____________________Page:' + str(page) + "-----end")
        # 下一页
        self.get_search_page(page=page + 1)

    def get_news_detail(self, page=1, offset=None):
        urls = []
        while True:
            urls = self.get_news_urls_from_db(page, offset)
            if len(urls) == 0:
                break
            print("#######Page: " + str(page) + "######")
            self.logger.log_now("Start get " + str(constant.LIMIT) + " news detail")
            data_list = []
            failed_news_urls = []
            index = 0
            for url in urls:
                print(str(index) + "###" + url)
                selenium_query.open_page(self.driver, url, sleep_time=2)
                page_data = self.get_detail_content(url)
                if page_data is None:
                    failed_news_urls.append({'source': self.source, 'url': url})
                else:
                    data_list.append(page_data)
                index = index + 1
            self.logger.log_now("Finish get " + str(constant.LIMIT) + " news detail")
            self.logger.log_now("Start insert news detail")
            db_obj = db.DB()
            if len(data_list) > 0:
                db_obj.insert_news(data_list)
            if len(failed_news_urls) > 0:
                db_obj.insert_failed_news_urls(failed_news_urls)
            db_obj.close_conn()
            self.logger.log_now("Finish insert news detail")
            page = page + 1

    def get_detail_content(self, url):
        # (source, url, title, content, date)
        page_data = {'source': self.source, 'url': url}
        news_title_area = self.driver.find_element_by_xpath('//div[@class="news_title_area"]')
        if news_title_area is None:
            print("news_title_area is None")
            return None
        title_article_h1 = selenium_custom.safe_find_elm(news_title_area, './/h1[@class="news_title"]')
        page_data['title'] = title_article_h1.text
        news_text_area = self.driver.find_element_by_xpath('//div[contains(@class, "news_text_area")]')
        if news_text_area is None:
            print("news_text_area is None")
            return None
        page_data['content'] = news_text_area.text
        news_date_p = selenium_custom.safe_find_elm(news_title_area, './/p[@class="news_date"]')
        if news_date_p is None:
            print("news_date_p is None")
            return None
        # Published : Dec. 28, 2023 - 10:11
        # Published : Sept. 30, 2023 - 20:50
        page_data['date'] = utils.koreaherald_date_string_to_datetime(news_date_p.text)
        return page_data

    def get_date(self, id, url):
        # (source, url, title, content, date)
        page_data = {'source': self.source, 'url': url, 'id': id}
        news_title_area = self.driver.find_element_by_xpath('//div[@class="news_title_area"]')
        if news_title_area is None:
            print("news_title_area is None")
            return None
        news_date_p = selenium_custom.safe_find_elm(news_title_area, './/p[@class="news_date"]')
        if news_date_p is None:
            print("news_date_p is None")
            return None
        # Published : Dec. 28, 2023 - 10:11
        # Published : Sept. 30, 2023 - 20:50
        page_data['date'] = utils.koreaherald_date_string_to_datetime(news_date_p.text)
        return page_data

    def get_news_urls_from_db(self, page, offset=None, where=None):
        self.logger.log_now("Start get news urls from db")
        db_obj = db.DB()
        if offset is None:
            offset = (page - 1) * constant.LIMIT
        if where is None:
            where = " WHERE source='" + self.source + "'"
        sql = "select * from " + db.table_news_urls + where + " LIMIT " + str(constant.LIMIT) \
              + " OFFSET " + str(offset)
        result = db_obj.query_results(sql)
        data = list(result['data'])
        urls = []
        for item in data:
            urls.append(item[2])
        db_obj.close_conn()
        self.logger.log_now("Finish get news urls from db")
        return urls

    def reprocess_date(self, page=1):
        where = " WHERE source='" + self.source + "' and date is null "
        db_obj = db.DB()
        news = db_obj.query_news(page=page, where=where)
        if len(news) == 0:
            return
        print('____________________Page:' + str(page) + "-----start")
        data_list = []
        for news_item in news:
            print(str(news_item['id']) + "###" + news_item['url'])
            selenium_query.open_page(self.driver, news_item['url'], sleep_time=2)
            page_data = self.get_date(news_item['id'], news_item['url'])
            data_list.append(page_data)
        db_obj.update_news_date(data_list)
        db_obj.close_conn()
        self.reprocess_date(page=page + 1)

    # ------ start to process the news ------
    def news_data_to_excel(self):
        sql = 'select * from news'
        excel_obj = excel.Excel()
        excel_obj.data_from_db_to_excel(sql, os.path.join(constant.ROOT_DIR, self.source), 'raw.xlsx')

    def filter_news_by_date_range(self):
        DataPreprocess().keep_news_in_date_range(os.path.join(constant.ROOT_DIR, self.source),
                                                 'raw.xlsx',
                                                 '2023-09-01 00:00:0', '2023-12-01 00:00:0',
                                                 os.path.join(constant.ROOT_DIR, self.source),
                                                 'news-0901-1130.xlsx')

    def remove_noise(self, source_filename, dest_filename):
        excel_obj = excel.Excel()
        df = excel_obj.read_excel(os.path.join(constant.ROOT_DIR, self.source), source_filename)
        df[constant.NEW_COL_NOISE_FLAG] = 0
        for idx, row in df.iterrows():
            content = row['内容'].lower()
            title = row['标题'].lower()
            if ('asian games' not in title and 'asian games' not in content) and \
                    ('asian para games' not in title and 'asian para games' not in content):
                df.loc[idx, constant.NEW_COL_NOISE_FLAG] = 1
        df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, self.source), dest_filename), index=False)

    def token_stopwords_lemma_spacy(self, source_filename, target_filename, source_col, target_col):
        self.logger.log_now('tokenization')
        DataPreprocess().token_stopwords_lemma_spacy(os.path.join(constant.ROOT_DIR, self.source),
                                                     source_filename,
                                                     os.path.join(constant.ROOT_DIR, self.source),
                                                     target_filename, remove_punct=False,
                                                     remove_line_break=True, remove_stop=True,
                                                     source_col=source_col, target_col=target_col)

    def word_cloud(self, source_filename, target_filename='wc_all.png', target_col=constant.NEW_COL_LEMMA_TEXT,
                   cluster_col=None, cluster=-1, max_words=30):
        DataAnalysis().generate_word_cloud(os.path.join(constant.ROOT_DIR, self.source),
                                           source_filename,
                                           os.path.join(constant.ROOT_DIR, self.source, target_filename),
                                           target_col=target_col, cluster_col=cluster_col, cluster=cluster,
                                           max_words=max_words)

    def tfidf_by_date_range(self, filename, start_date, end_date, target_col=constant.NEW_COL_LEMMA_TEXT,
                            target_filename_prefix='tfidf-'):
        X, tfidf_df = DataAnalysis().tf_idf(os.path.join(constant.ROOT_DIR, self.source),
                                            filename,
                                            start_date=start_date, end_date=end_date,
                                            target_col=target_col)
        tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, self.source),
                                       target_filename_prefix + start_date.split(' ')[0]
                                       + '-' + end_date.split(' ')[0] + '.xlsx'), index=False)

    def tfidf_by_cluster(self, filename, target_col=constant.NEW_COL_LEMMA_TEXT, cluster_col=None, cluster=-1):
        X, tfidf_df = DataAnalysis().tf_idf(os.path.join(constant.ROOT_DIR, self.source),
                                            filename,
                                            start_date=None, end_date=None,
                                            target_col=target_col, cluster_col=cluster_col, cluster=cluster)
        tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, self.source),
                                       'tfidf-' + cluster_col + '-' + str(cluster) + '.xlsx'), index=False)

    def tfidf_by_cluster_stance(self, filename, target_col=constant.NEW_COL_LEMMA_TEXT, cluster_col=None, cluster=-1, stance=None):
        X, tfidf_df = DataAnalysis().tf_idf(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT),
                                            filename,
                                            start_date=None, end_date=None,
                                            target_col=target_col, cluster_col=cluster_col, cluster=cluster, stance=stance)
        tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT),
                                       'tfidf-' + cluster_col + '-' + str(cluster) + '-' + stance + '.xlsx'), index=False)

    def tokenization_nltk(self):
        DataPreprocess().tokenization_nltk(os.path.join(constant.ROOT_DIR, self.source),
                                           'news-0901-1130.xlsx', os.path.join(constant.ROOT_DIR, self.source),
                                           'news-0901-1130-tokenization-nltk.xlsx')

    def clustering_kmeans(self, target_col, k, type=1):
        if type == 1:
            DataAnalysis().kmeans(os.path.join(constant.ROOT_DIR, self.source), 'news-0901-1130-tokenization2.xlsx',
                                  target_col=target_col, k=k, cluster_col_name=constant.NEW_COL_TITLE_CLUSTER,
                                  dest_file_dir=os.path.join(constant.ROOT_DIR, self.source),
                                  dest_filename='news-0901-1130-tokenization2.xlsx')
        elif type == 2:
            DataAnalysis().kmeans(os.path.join(constant.ROOT_DIR, self.source), 'news-0901-1130-tokenization2.xlsx',
                                  target_col=target_col, k=k, cluster_col_name=constant.NEW_COL_CONTENT_CLUSTER,
                                  dest_file_dir=os.path.join(constant.ROOT_DIR, self.source),
                                  dest_filename='news-0901-1130-tokenization2.xlsx')
        elif type == 3:
            DataAnalysis().kmeans(os.path.join(constant.ROOT_DIR, self.source), 'news-0901-1130-tokenization2-3.xlsx',
                                  target_col=target_col, k=k, cluster_col_name=constant.NEW_COL_CONTENT_CLUSTER,
                                  dest_file_dir=os.path.join(constant.ROOT_DIR, self.source),
                                  dest_filename='news-0901-1130-tokenization2-3.xlsx')

    def context_of_keyword(self, keyword, i, cluster=2):
        DataAnalysis().context_of_keyword(os.path.join(constant.ROOT_DIR, self.source),
                                          'news-0901-1130-tokenization2.xlsx', keyword,
                                          os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT,
                                                       'cluster_' + str(cluster)),
                                          'context_of_[' + keyword + '].xlsx',
                                          gap=i, cluster=cluster)

    def get_top_keywords_from_tfidf(self, source_filename, top=20, pos_list=['NOUN', 'PROPN'], skip_words=[]):
        excel_obj = excel.Excel()
        df = excel_obj.read_excel(os.path.join(constant.ROOT_DIR, self.source), source_filename)
        keywords = []
        for idx, row in df.iterrows():
            if idx >= top:
                break
            keyword = row['word']
            pos = row[constant.COL_POS]
            if keyword in skip_words:
                continue
            if pos not in pos_list:
                continue
            keywords.append(keyword)
        return keywords

    def context_of_keyword_top_from_tfidf(self, keywords, cluster=2):
        for keyword in keywords:
            self.context_of_keyword(keyword, i=5, cluster=cluster)

    def statistics_of_keyword_context_by_pos(self, keywords, cluster=2, stat_poses=['ADJ', 'VERB', 'NOUN', 'ADV']):
        for keyword in keywords:
            for stat_pos in stat_poses:
                kh.statistics_of_keyword_context(keyword, stat_pos, cluster=cluster)

    # POS: ADJ, VERB, NOUN, ADV
    def statistics_of_keyword_context(self, keyword, pos, cluster):
        excel_obj = excel.Excel()
        df = excel_obj.read_excel(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT,
                                               'cluster_' + str(cluster)), 'context_of_[' + keyword + '].xlsx')
        df = df[df[constant.COL_POS] == pos]
        contexts = {}
        for idx, row in df.iterrows():
            context = row[constant.COL_CONTEXT]
            if context not in contexts:
                contexts[context] = 0
            contexts[context] = contexts[context] + 1
        data = {
            constant.COL_CONTEXT: list(contexts.keys()),
            constant.COL_FREQUENCY: list(contexts.values())
        }
        context_df = pd.DataFrame(data).sort_values(by=constant.COL_FREQUENCY, ascending=False, ignore_index=True)
        context_df[constant.COL_KEYWORD] = keyword
        context_df[constant.COL_POS] = pos
        context_df.to_excel(
            os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT, 'cluster_' + str(cluster),
                         'stat_context_of_[' + keyword + ']_by_' + pos + '.xlsx'), index=False)

    def word_pos(self, source_filename, cluster, context_filename):
        DataAnalysis().word_pos_from_context_file(os.path.join(constant.ROOT_DIR, self.source), source_filename,
                                                  os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT,
                                                               'cluster_' + str(cluster)),
                                                  context_filename)

    def word_pos_of_sentiment(self, source_filename, cluster, context_filename, overwrite):
        DataAnalysis().word_pos_from_context_file(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT),
                                                  source_filename,
                                                  os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT,
                                                               'cluster_' + str(cluster)),
                                                  context_filename, overwrite=overwrite)

    def combine_statistics_of_keyword_top(self, keywords, cluster, top=20, stat_poses=['ADJ', 'VERB', 'NOUN', 'ADV']):
        excel_obj = excel.Excel()
        context_list = []
        frequency_list = []
        keyword_list = []
        pos_list = []
        for keyword in keywords:
            for pos in stat_poses:
                df = excel_obj.read_excel(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT,
                                                       'cluster_' + str(cluster)),
                                          'stat_context_of_[' + keyword + ']_by_' + pos + '.xlsx')
                for idx, row in df.iterrows():
                    if idx >= top:
                        break
                    context_list.append(row[constant.COL_CONTEXT])
                    frequency_list.append(row[constant.COL_FREQUENCY])
                    keyword_list.append(row[constant.COL_KEYWORD])
                    pos_list.append(row[constant.COL_POS])
            data = {
                constant.COL_CONTEXT: context_list,
                constant.COL_FREQUENCY: frequency_list,
                constant.COL_KEYWORD: keyword_list,
                constant.COL_POS: pos_list,
            }
            all_df = pd.DataFrame(data)
            all_df.to_excel(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_CONTEXT, 'cluster_' + str(cluster),
                                         'top_' + str(top) + 'all_pos_stat_context_of_[' + keyword + '].xlsx'), index=False)

    def appraisal_attitude(self, source_filename, appraisal_attitude_filename, target_filename):
        DataAnalysis().appraisal_attitude_analysis(os.path.join(constant.ROOT_DIR, self.source), source_filename,
                                                   os.path.join(constant.ROOT_DIR, self.source),
                                                   appraisal_attitude_filename,
                                                   os.path.join(constant.ROOT_DIR, self.source,
                                                                constant.DIR_APPRAISAL_ATTITUDE), target_filename,)

    def lemma_text_to_sentence(self, source_filename, dest_filename):
        DataAnalysis().lemma_text_to_sentences(os.path.join(constant.ROOT_DIR, self.source), source_filename,
                                                   os.path.join(constant.ROOT_DIR, self.source), dest_filename)

    def sentiment_analysis_by_textblob(self, source_filename, dest_filename):
        DataAnalysis().sentiment_analysis_by_textblob(os.path.join(constant.ROOT_DIR, self.source), source_filename,
                                               os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT), dest_filename,
                                                      col_name=constant.COL_SENTENCE)

    def plot_sentiment_result(self, source_filename, dest_filename, cluster):
        label = ''
        if cluster == 0:
            label = '"政治话题"主题的情感倾向'
        elif cluster == 1:
            label = '"比赛战报"主题的情感倾向'
        elif cluster == 2:
            label = '"杭州印象"主题的情感倾向'
        DataAnalysis().plot_sentiment_result(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT), source_filename,
                                               os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT), dest_filename,
                                                      cluster=cluster, label=label)

    def plot_sentence_count_by_cluster(self, source_filename, dest_filename):
        DataAnalysis().plot_sentence_count_by_cluster(os.path.join(constant.ROOT_DIR, self.source, constant.DIR_SENTIMENT), source_filename,
                                                      os.path.join(constant.ROOT_DIR, self.source,
                                                                   constant.DIR_SENTIMENT), dest_filename,
                                                      )

if __name__ == "__main__":
    kh = KoreaHerald(driver_needed=False)
    # kh.get_search_page(page=1)
    # kh.get_news_detail(page=62, offset=None)
    # kh.reprocess_date(page=1)
    # kh.news_data_to_excel()
    # kh.filter_news_by_date_range()
    # kh.token_stopwords_lemma_spacy('news-0901-1130.xlsx', 'news-0901-1130-tokenization2.xlsx')
    # kh.word_cloud()
    # kh.tfidf_by_date_range('news-0901-1130-tokenization2.xlsx', '2023-09-01 00:00:0', '2023-09-23 00:00:0',
    #                        target_col=constant.NEW_COL_TITLE_LEMMA_TEXT, target_filename_prefix='tfidf-title-')  # 前
    # kh.tfidf_by_date_range('news-0901-1130-tokenization2.xlsx', '2023-09-23 00:00:0', '2023-10-29 00:00:0',
    #                        target_col=constant.NEW_COL_TITLE_LEMMA_TEXT, target_filename_prefix='tfidf-title-')  # 中
    # kh.tfidf_by_date_range('news-0901-1130-tokenization2.xlsx', '2023-10-29 00:00:0', '2023-12-01 00:00:0',
    #                        target_col=constant.NEW_COL_TITLE_LEMMA_TEXT, target_filename_prefix='tfidf-title-')  # 后
    # kh.tokenization_nltk()
    # kh.word_cloud('news-0901-1130-tokenization2-1.xlsx', target_filename='wc_前.png')
    # kh.token_stopwords_lemma_spacy('news-0901-1130-tokenization2.xlsx', 'news-0901-1130-tokenization2.xlsx',
    #                              source_col='标题', target_col=constant.NEW_COL_TITLE_LEMMA_TEXT)
    # kh.clustering_kmeans(target_col=constant.NEW_COL_TITLE_LEMMA_TEXT, k=3, type=1)
    # kh.clustering_kmeans(target_col=constant.NEW_COL_LEMMA_TEXT, k=3, type=2)
    # kh.remove_noise('news-0901-1130-tokenization2.xlsx', 'news-0901-1130-tokenization2.xlsx')
    # kh.tfidf_by_cluster('news-0901-1130-tokenization2.xlsx', target_col=constant.NEW_COL_TITLE_LEMMA_TEXT,
    #                        cluster_col=constant.NEW_COL_TITLE_CLUSTER, cluster=2)
    # kh.word_cloud('news-0901-1130-tokenization2.xlsx',
    #               target_filename='wc_' + constant.NEW_COL_CONTENT_CLUSTER + '-' + str(2) + '.png',
    #               target_col=constant.NEW_COL_LEMMA_TEXT, cluster_col=constant.NEW_COL_CONTENT_CLUSTER, cluster=2)
    # kh.clustering_kmeans(target_col=constant.NEW_COL_LEMMA_TEXT, k=5, type=3)
    # kh.word_pos('tfidf-根据内容聚类-0.xlsx', 0, 'stat_context_of_[xi]_by_VERB.xlsx')
    # kh.word_pos('tfidf-根据内容聚类-0.xlsx', 0, 'stat_context_of_[xi]_by_ADJ.xlsx')
    # kh.word_pos('tfidf-根据内容聚类-0.xlsx', 0, 'stat_context_of_[xi]_by_ADV.xlsx')
    # kh.word_pos('tfidf-根据内容聚类-0.xlsx', 0, 'stat_context_of_[xi]_by_NOUN.xlsx')
    # kh.word_pos('tfidf-根据内容聚类-1.xlsx', 1, 'context_of_[korea].xlsx')
    # kh.word_pos('tfidf-根据内容聚类-2.xlsx', 2, 'context_of_[athlete].xlsx')
    # keywords = ['games', 'hangzhou']
    # poses = ['ADJ', 'VERB', 'NOUN', 'ADV']
    # for keyword in keywords:
    #     kh.context_of_keyword(keyword, i=5, cluster=1)
    # for pos in poses:
    #     kh.statistics_of_keyword_context(keyword, pos)
    # cluster_2_skip_words = ['oct']
    # cluster_1_skip_words = []
    # cluster_0_skip_words = []
    # keywords = kh.get_top_keywords_from_tfidf('tfidf-根据内容聚类-0.xlsx', skip_words=cluster_0_skip_words)
    # kh.context_of_keyword_top_from_tfidf(keywords, cluster=0)
    # kh.statistics_of_keyword_context_by_pos(keywords, cluster=0)
    # kh.combine_statistics_of_keyword_top(keywords, cluster=0)
    # kh.appraisal_attitude('tfidf-根据内容聚类-0.xlsx', 'appraisal-attitude.xlsx', 'appraisal-attitude-cluster-0.xlsx')
    # kh.appraisal_attitude('tfidf-根据内容聚类-1.xlsx', 'appraisal-attitude.xlsx', 'appraisal-attitude-cluster-1.xlsx')
    # kh.appraisal_attitude('tfidf-根据内容聚类-2.xlsx', 'appraisal-attitude.xlsx', 'appraisal-attitude-cluster-2.xlsx')
    # kh.lemma_text_to_sentence('news-0901-1130-tokenization-clusters.xlsx', 'doc_sentences_by_clusters.xlsx')
    # kh.sentiment_analysis_by_textblob('doc_sentences_by_clusters.xlsx', 'doc_sentence_sentiment_by_clusters.xlsx')
    # kh.plot_sentiment_result('doc_sentence_sentiment_by_clusters.xlsx', 'doc_sentence_sentiment_cluster0.png', 0)
    # kh.plot_sentiment_result('doc_sentence_sentiment_by_clusters.xlsx', 'doc_sentence_sentiment_cluster1.png', 1)
    # kh.plot_sentiment_result('doc_sentence_sentiment_by_clusters.xlsx', 'doc_sentence_sentiment_cluster2.png', 2)
    # kh.plot_sentence_count_by_cluster('doc_sentence_sentiment_by_clusters.xlsx', 'doc_sentence_count_by_cluster.png')
    # kh.tfidf_by_cluster_stance('doc_sentence_sentiment_by_clusters.xlsx', target_col=constant.COL_SENTENCE,
    #                        cluster_col=constant.NEW_COL_CONTENT_CLUSTER, cluster=2, stance='积极')
    # kh.word_pos_of_sentiment('tfidf-根据内容聚类-2-积极.xlsx', 2, 'context_of_[event].xlsx', overwrite=False)
    # kh.word_pos_of_sentiment('tfidf-根据内容聚类-2-消极.xlsx', 2, 'context_of_[city].xlsx', overwrite=False)