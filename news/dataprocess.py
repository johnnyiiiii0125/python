import os
import file
import pandas as pd
from common import constant
from excel import Excel
from common import utils
import matplotlib.pyplot as plot
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
# from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from pattern.text.en import sentiment
import seaborn as sns
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import spacy

# 处理plt中文显示问题
plot.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文标签
plot.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

TFIDF_STOPWORDS = ['bbc', 'tokyo', 'olympic', 'olympics', 'cambage', '2020',
                   'brisbane', 'watch', 'coverage', 'app', 'iplayer', 'website']


# 2021年7月23日开幕、8月8日闭幕
# 数据时间：'2021-07-16 00:00:0', '2021-08-09 00:00:0'

class DataAnalysis:
    def __init__(self):
        self.excel_obj = Excel()
        self.tfidf_vectorizer = None
        self.tfidf_X = None
        self.lda = None
        self.nlp = spacy.load('en_core_web_lg')

    def generate_word_cloud(self, source_file_dir, source_filename, saved_file_path, news_source=None,
                            start_date=None, end_date=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
        # text = ' '.join(content for content in df[constant.NEW_COL_LEMMA_TEXT])
        docs = [content for content in df[constant.NEW_COL_LEMMA_TEXT]]
        vectorizer = TfidfVectorizer(stop_words=TFIDF_STOPWORDS)
        vecs = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()
        dense = vecs.todense()
        lst1 = dense.tolist()
        df_tfidf = pd.DataFrame(lst1, columns=feature_names)
        # Generate word cloud
        word_cloud = WordCloud(
            width=3000,
            height=2000,
            random_state=1,
            background_color="white",
            # colormap="Pastel1",
            collocations=False,
            max_words=200,
        ).generate_from_frequencies(df_tfidf.T.sum(axis=1))
        plot.imshow(word_cloud)
        plot.axis('off')
        # plot.show()
        word_cloud.to_file(saved_file_path)

    def generate_word_cloud_of_sentiment(self, source_file_dir, source_filename, saved_file_path,
                                         classification, more_stopwords=None, word_class='ADJ'):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if classification is not None:
            df = df[df['立场'] == classification]
        raw_docs = [content for content in df['语境']]
        docs = []
        for content in raw_docs:
            doc = self.nlp(content)
            doc_remain = []
            for token in doc:
                print(token.text + '-----' + token.pos_)
                if token.pos_ == word_class:
                    doc_remain.append(token.text)
            docs.append(' '.join(doc_remain))
        stopwords = []
        stopwords.extend(TFIDF_STOPWORDS)
        if more_stopwords is not None:
            stopwords.extend(more_stopwords)
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        vecs = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()
        dense = vecs.todense()
        lst1 = dense.tolist()
        df_tfidf = pd.DataFrame(lst1, columns=feature_names)
        # Generate word cloud
        word_cloud = WordCloud(
            width=3000,
            height=2000,
            random_state=1,
            background_color="white",
            # colormap="Pastel1",
            collocations=False,
            max_words=200,
        ).generate_from_frequencies(df_tfidf.T.sum(axis=1))
        plot.imshow(word_cloud)
        plot.axis('off')
        word_cloud.to_file(saved_file_path)

    def generate_word_cloud_by_date(self, source_file_dir, source_filename, news_source=None):
        start_date = '2021-7-16 00:00:0'
        stop_date = '2021-8-9 00:00:0'
        dest_filename0 = 'wc'
        if news_source is not None:
            dest_filename0 += '_' + news_source
        while True:
            print(start_date)
            dest_filename = dest_filename0 + '_' + start_date.split(' ')[0] + '.png'
            next_date = utils.next_day(start_date)
            self.generate_word_cloud(source_file_dir, source_filename,
                                     os.path.join(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
                                                               constant.DIR_WORDCLOUD),
                                                  dest_filename), news_source=news_source,
                                     start_date=start_date, end_date=next_date)
            if next_date == stop_date:
                break
            start_date = next_date

    def total_word_count(self, source_file_dir, source_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        all_unique_words = []
        for idx, row in df.iterrows():
            content = row[constant.NEW_COL_LEMMA_TEXT]
            tokens = content.split(' ')
            print('doc####', idx)
            for token in tokens:
                token = token.strip()
                if token not in all_unique_words:
                    all_unique_words.append(token)
        print(len(all_unique_words))
        return all_unique_words

    # def lda_model(self, source_file_dir, source_filename, news_source=None):
    #     df = self.excel_obj.read_excel(source_file_dir, source_filename)
    #     if news_source is not None:
    #         df = df[df['媒体'] == news_source]
    #     docs = [content.split(' ') for content in df[NEW_COL_LEMMA_TEXT]]
    #     dictionary = corpora.Dictionary(docs)
    #     print(dictionary)
    #     # print(dictionary.token2id)
    #     corpus = [dictionary.doc2bow(doc) for doc in docs]
    #     print(len(corpus))
    #     ldamodel = models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10)
    #     print(ldamodel.print_topics(num_topics=5, num_words=10))
    #     return ldamodel

    def tf_idf(self, source_file_dir, source_filename, news_source=None, start_date=None, end_date=None):
        """
        TF:

        IDF:

        """
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
        docs = [content for content in df[constant.NEW_COL_LEMMA_TEXT]]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=TFIDF_STOPWORDS)
        self.tfidf_X = self.tfidf_vectorizer.fit_transform(docs)
        data = {
            'word': self.tfidf_vectorizer.get_feature_names(),
            'tfidf': self.tfidf_X.toarray().sum(axis=0).tolist()
        }
        tfidf_df = pd.DataFrame(data).sort_values(by="tfidf", ascending=False, ignore_index=True)
        return self.tfidf_X, tfidf_df

    def tf_idf_by_date(self, source_file_dir, source_filename, news_source=None):
        start_date = '2021-7-16 00:00:0'
        stop_date = '2021-8-9 00:00:0'
        dest_filename0 = 'tfidf'
        if news_source is not None:
            dest_filename0 += '_' + news_source
        while True:
            print(start_date)
            dest_filename = dest_filename0 + '_' + start_date.split(' ')[0] + '.xlsx'
            next_date = utils.next_day(start_date)
            X, tfidf_df = self.tf_idf(source_file_dir, source_filename, news_source=news_source,
                                      start_date=start_date, end_date=next_date)
            tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
                                           dest_filename), index=True)
            if next_date == stop_date:
                break
            start_date = next_date

    def clustering_kmeans(self, data, k):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=20000, random_state=0)
        y_means = kmeans.fit_predict(data)
        return y_means

    '''
        LDA model 对于本次数据源不适用
    '''

    def lda_model(self, n_topics=10):
        self.lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                             learning_method='batch',
                                             learning_offset=100,
                                             #                                 doc_topic_prior=0.1,
                                             #                                 topic_word_prior=0.01,
                                             random_state=0)
        self.lda.fit(self.tfidf_X)
        return self.lda

    def print_lda_top_words(self, model, feature_names, n_top_words):
        tword = []
        tword2 = []
        tword3 = []
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topic_pro = [str(round(topic[i], 3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  # (round(topic[i],3))
            tword.append(topic_w)
            tword2.append(topic_pro)
            print(" ".join(topic_w))
            print(" ".join(topic_pro))
            print(' ')
            word_pro = dict(zip(topic_w, topic_pro))
            tword3.append(word_pro)
        return tword3

    '''
    find the context of the keyword
    '''

    def context_by_keyword(self, source_file_dir, source_filename, keyword, dest_file_dir, context_length=10,
                           news_source=None, allocation=None, start_date=None, end_date=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        dest_filename = keyword
        if allocation is not None:
            dest_filename = keyword + '_' + "_".join(allocation)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
            dest_filename += '_' + news_source
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
            dest_filename += '_' + start_date.split(' ')[0] + '_' + end_date.split(' ')[0]
        dest_filename += '.xlsx'
        docs_keyword_contexts = {}
        doc_keyword_context_ids = []
        doc_keyword_context_dates = []
        doc_keyword_context_content = []
        for idx, row in df.iterrows():
            content = row[constant.NEW_COL_LEMMA_TEXT]
            id = row[constant.COL_ID]
            publish_date = row['发布日期']
            tokens = content.split(' ')
            # print('doc####', id)
            for i, token in enumerate(tokens):
                token = token.strip()
                if keyword in token:
                    pre_i = i - context_length
                    if pre_i < 0:
                        pre_i = 0
                    suf_i = i + context_length
                    keyword_context = ''
                    for j, token2 in enumerate(tokens):
                        if pre_i <= j <= suf_i:
                            keyword_context += token2 + ' '
                        if j > suf_i:
                            break
                    if allocation is None:
                        doc_keyword_context_ids.append(id)
                        doc_keyword_context_dates.append(publish_date)
                        doc_keyword_context_content.append(keyword_context.strip())
                    else:
                        all_allocation_is_in = True
                        for alloc in allocation:
                            if alloc not in keyword_context.strip():
                                all_allocation_is_in = False
                                break
                        if all_allocation_is_in:
                            doc_keyword_context_ids.append(id)
                            doc_keyword_context_dates.append(publish_date)
                            doc_keyword_context_content.append(keyword_context.strip())
        docs_keyword_contexts['文档序号'] = doc_keyword_context_ids
        docs_keyword_contexts['语境'] = doc_keyword_context_content
        docs_keyword_contexts['发布日期'] = doc_keyword_context_dates
        context_df = pd.DataFrame(docs_keyword_contexts, columns=['文档序号', '语境', '发布日期'])
        context_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def lines_of_keyword(self, source_file_dir, source_filename, keyword, dest_file_dir, line_length=1,
                         news_source=None, allocation=None, start_date=None, end_date=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        dest_filename = keyword
        if allocation is not None:
            dest_filename = keyword + '_' + "_".join(allocation)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
            dest_filename += '_' + news_source
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
            dest_filename += '_' + start_date.split(' ')[0] + '_' + end_date.split(' ')[0]
        dest_filename += '.xlsx'
        docs_keyword_contexts = {}
        doc_keyword_context_ids = []
        doc_keyword_context_dates = []
        doc_keyword_context_content = []
        for idx, row in df.iterrows():
            content = row[constant.NEW_COL_LEMMA_TEXT]
            id = row[constant.COL_ID]
            publish_date = row['发布日期']
            lines = content.split('.')
            print('doc####', id)
            for i, line in enumerate(lines):
                line = line.strip()
                context_lines = []
                if keyword in line:
                    pre_i = i - line_length
                    if pre_i < 0:
                        pre_i = 0
                    suf_i = i + line_length
                    all_allocation_is_in = True
                    if allocation is not None:
                        for alloc in allocation:
                            if alloc not in line.strip():
                                all_allocation_is_in = False
                                break
                    if all_allocation_is_in:
                        for j, line2 in enumerate(lines):
                            if pre_i <= j <= suf_i:
                                context_lines.append(line2.strip())
                            if j > suf_i:
                                break
                        doc_keyword_context_ids.append(id)
                        doc_keyword_context_dates.append(publish_date)
                        doc_keyword_context_content.append(".".join(context_lines))
        docs_keyword_contexts['文档序号'] = doc_keyword_context_ids
        docs_keyword_contexts['语境'] = doc_keyword_context_content
        docs_keyword_contexts['发布日期'] = doc_keyword_context_dates
        context_df = pd.DataFrame(docs_keyword_contexts, columns=['文档序号', '语境', '发布日期'])
        context_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def time_to_date(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df['发布日期'] = df['发布时间'].dt.date
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def get_news_count_by_date(self, source_file_dir, source_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df1 = df.groupby('发布日期').count()
        df1['id'].plot()
        plot.title('奥运赛事期间三大媒体每日共计报道的新闻数量')
        plot.xlabel('发布日期')
        plot.ylabel('新闻数')
        plot.grid()
        plot.show()

    def sentiment_analysis_by_pattern(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df['极性'] = 0
        df['主观性'] = 0
        for idx, row in df.iterrows():
            context = row['语境']
            id = row['文档序号']
            result = sentiment(context)
            df.loc[idx, '极性'] = result[0]
            df.loc[idx, '主观性'] = result[1]
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def sentiment_analysis_by_textblob(self, source_file_dir, source_filename, dest_file_dir, dest_filename, ignore=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df['极性'] = 0
        df['主观性'] = 0
        df['立场'] = ''
        for idx, row in df.iterrows():
            context = row['语境']
            if ignore is not None:
                for word in ignore:
                    context = context.replace(word, '')
            result = TextBlob(context)
            df.loc[idx, '极性'] = result.sentiment.polarity
            df.loc[idx, '主观性'] = result.sentiment.subjectivity
        df.loc[df['极性'] > 0, '立场'] = '积极'
        df.loc[df['极性'] < 0, '立场'] = '消极'
        df.loc[df['极性'] == 0, '立场'] = '中立'
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def sentiment_analysis_by_textblob_nba(self, source_file_dir, source_filename, dest_file_dir, dest_filename, ignore=None):
        # anyway   try it!!!
        # not suitable for us....it is based on movie reviews
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df['p_pos'] = 0
        df['p_neg'] = 0
        df['立场'] = ''
        df['分类'] = ''
        for idx, row in df.iterrows():
            print(idx)
            context = row['语境']
            if ignore is not None:
                for word in ignore:
                    context = context.replace(word, '')
            result = TextBlob(context, analyzer=NaiveBayesAnalyzer())
            classification = result.sentiment.classification
            df.loc[idx, 'p_pos'] = result.sentiment.p_pos
            df.loc[idx, 'p_neg'] = result.sentiment.p_neg
            df.loc[idx, '分类'] = classification
        # df.loc[df['极性'] > 0, '立场'] = '积极'
        # df.loc[df['极性'] < 0, '立场'] = '消极'
        # df.loc[df['极性'] == 0, '立场'] = '中立'
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def plot_sentiment_result(self, source_file_dir, source_filename, dest_file_dir=None, dest_filename=None,
                              label='极性'):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        # sns.displot(df["立场"], height=5, aspect=1.8)
        # df.groupby('立场').count().plot(kind='pie', y='立场', autopct='%1.0f%%')
        plot.pie(x=df['立场'].value_counts(), labels=df['立场'].value_counts().index, autopct='%1.0f%%')
        plot.title(label)
        plot.savefig(os.path.join(dest_file_dir, dest_filename))


# word class: ADJ, NOUN, PROPN, ADV, VERB
word_class = 'NOUN'
DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
                                                'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx',
                                                os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
                                                             constant.DIR_WORDCLOUD, 'ceremony_'+word_class.lower()+'.png'),
                                                None, more_stopwords=None, word_class=word_class)
print("-----------------------------------------------")
sentiment_wc_stopwords = ['japanese', 'australian', 'israeli', 'palestinian']
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'ceremony_pos.png'),
#                                                 '积极', more_stopwords=sentiment_wc_stopwords)
# print("-----------------------------------------------")
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'ceremony_neg.png'),
#                                                 '消极', more_stopwords=sentiment_wc_stopwords)

# sentiment_wc_stopwords.extend(['positive', 'new'])
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'covid_case_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'covid_case_neg.png'),
#                                                 '消极', more_stopwords=sentiment_wc_stopwords)
# print("-----------------------------------------------")
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'covid_case_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'covid_case_pos.png'),
#                                                 '积极', more_stopwords=sentiment_wc_stopwords)
# print("-----------------------------------------------")
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'covid_positive_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'covid_positive_neg.png'),
#                                                 '消极', more_stopwords=sentiment_wc_stopwords)
# print("-----------------------------------------------")
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'covid_positive_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'covid_positive_pos.png'),
#                                                 '积极', more_stopwords=sentiment_wc_stopwords)
# plot sentiment
# DataAnalysis().plot_sentiment_result(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                      os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_textblob_pie.png',
#                                      label='7月16日-20日三大媒体对新冠肺炎的情感态度'
#                                      )
# DataAnalysis().plot_sentiment_result(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_case_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                      os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_case_textblob_pie.png',
#                                      label='7月16日-20日三大媒体对新冠肺炎案例的情感态度'
#                                      )
# DataAnalysis().plot_sentiment_result(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_positive_2021-7-16_2021-7-21_textblob_sentiment.xlsx',
#                                      os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'covid_positive_textblob_pie.png',
#                                      label='7月16日-20日三大媒体对新冠肺炎阳性的情感态度'
#                                      )
# DataAnalysis().plot_sentiment_result(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx',
#                                       os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                      'ceremony_textblob_pie.png',
#                                      label='7月22日-23日三大媒体对开幕式的情感态度'
#                                      )
# sentiment analysis
# DataAnalysis().sentiment_analysis_by_pattern(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'ceremony_2021-7-22_2021-7-24.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'ceremony_2021-7-22_2021-7-24_sentiment.xlsx')

# DataAnalysis().sentiment_analysis_by_textblob(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'ceremony_2021-7-22_2021-7-24.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                   'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx')

# DataAnalysis().sentiment_analysis_by_textblob_nba(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'ceremony_2021-7-22_2021-7-24.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                   'ceremony_2021-7-22_2021-7-24_textblob_nba_sentiment.xlsx')

# DataAnalysis().sentiment_analysis_by_textblob(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'covid_2021-7-16_2021-7-21.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                   'covid_2021-7-16_2021-7-21_textblob_sentiment.xlsx', ignore=['positive', 'new'])

# DataAnalysis().sentiment_analysis_by_textblob(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'covid_case_2021-7-16_2021-7-21.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                   'covid_case_2021-7-16_2021-7-21_textblob_sentiment.xlsx', ignore=['positive', 'new'])
#
# DataAnalysis().sentiment_analysis_by_textblob(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT),
#                                   'covid_positive_2021-7-16_2021-7-21.xlsx',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                   'covid_positive_2021-7-16_2021-7-21_textblob_sentiment.xlsx', ignore=['positive', 'new'])

# DataAnalysis().get_news_count_by_date(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                   'news_token_lemma-0716-0808.xlsx')


# contexts
# DataAnalysis().context_by_keyword(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                    'news_token_lemma-0716-0808.xlsx', 'covid',
#                                   os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT), 10,
#                                   news_source=None, allocation=None,
#                                   start_date='2021-7-16 00:00:0', end_date='2021-7-21 00:00:0')
# DataAnalysis().lines_of_keyword(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                 'news_token_lemma-0716-0808.xlsx', 'ceremony',
#                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT), 1,
#                                 news_source=None, allocation=None,
#                                 start_date='2021-7-22 00:00:0', end_date='2021-7-24 00:00:0')
# DataAnalysis().lines_of_keyword(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                 'news_token_lemma-0716-0808.xlsx', 'covid',
#                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_CONTEXT), 1,
#                                 news_source=None, allocation=['positive'],
#                                 start_date='2021-7-16 00:00:0', end_date='2021-7-21 00:00:0')
# tf idf  && kmeans
# dataAnalysis = DataAnalysis()
# X, tfidf_df = dataAnalysis.tf_idf(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                   'news_token_lemma-0716-0808.xlsx', news_source=None)
# tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                'tfidf_' + constant.SOURCE_BBC + '.xlsx'), index=True)

# by date
# dataAnalysis = DataAnalysis()
# dataAnalysis.tf_idf_by_date(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                   'news_token_lemma-0716-0808.xlsx', news_source=None)

# lda = dataAnalysis.lda_model(2)
# dataAnalysis.print_lda_top_words(lda, dataAnalysis.tfidf_vectorizer.get_feature_names(), 20)
# y_means = dataAnalysis.clustering_kmeans(X, 3)
# print(len(y_means))
# print(len(tfidf_df))

# print(DataPreprocess().nlp.Defaults.stop_words)

# DataAnalysis().lda_model(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx', news_source='cnn')

# get all words count: 34621
# DataAnalysis().total_word_count(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx')

# generate word cloud
# DataAnalysis().generate_word_cloud(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx',
#             os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, 'wc_all.png')
#                                    # , news_source='abc'
#                                    )

# generate word cloud by date
# DataAnalysis().generate_word_cloud_by_date(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                            'news_token_lemma-0716-0808.xlsx'
#                                            , news_source=None
#                                            )

# DataAnalysis().time_to_date(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma-0716-0808.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma-0716-0808.xlsx')

# 5. word count
# word_counts_all = {}
# if not os.path.exists(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.FILE_KEYWORDS_BY_SOURCE)):
#     word_counts_all = DataPreprocess().word_counts_all_dataset(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                   'news_token_lemma.xlsx', by_news_source=True)
#     file.dict_to_json_file(word_counts_all, os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), constant.FILE_KEYWORDS_BY_SOURCE)
# else:
#     word_counts_all = file.read_file_to_json(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), constant.FILE_KEYWORDS_BY_SOURCE)
# sorted_in_pos = DataPreprocess().sort_word_counts(word_counts_all, 20, by_news_source=True)
# file.dict_to_json_file(sorted_in_pos, os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'keywords_source_top20.json')

# 4. tokenization remove stop words, lemma
# DataPreprocess().token_stopwords_lemma_spacy(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx')
# # 4.2 get noun trunks from doc
# DataPreprocess().generate_noun_chunks(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx')

# 3.2 将内容中的换行符去除
# DataPreprocess().remove_content_linebreak(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx')

# 3.1 去除噪音数据
# DataPreprocess().remove_noise(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_token_lemma.xlsx')

# 3.5 保留某个时间段
# DataPreprocess().keep_news_in_date_range(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                          'news_token_lemma.xlsx',
#                                          '2021-07-16 00:00:0', '2021-08-09 00:00:0',
#                                          os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                          'news_token_lemma-0716-0808.xlsx')

# 3.只保留2021年的新闻
# DataPreprocess().keep_news_in_2021(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_not_null.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_2021.xlsx')

# 2. 去除content为空的新闻
# DataPreprocess().remove_null_rows(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_raw.xlsx', os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_not_null.xlsx')

# 1.将news表所有数据导出
# Excel().data_from_db_to_excel('select * from news',
#                               os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                               'news_raw.xlsx')
