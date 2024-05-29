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
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from pattern.text.en import sentiment
import seaborn as sns
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import spacy
import numpy as np
from sklearn.decomposition import PCA

# 处理plt中文显示问题
plot.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文标签
plot.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

# TFIDF_STOPWORDS = ['bbc', 'tokyo', 'olympic', 'olympics', 'cambage', '2020',
#                   'brisbane', 'watch', 'coverage', 'app', 'iplayer', 'website']
TFIDF_STOPWORDS = []

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
                            start_date=None, end_date=None, target_col=constant.NEW_COL_LEMMA_TEXT,
                            cluster_col=None, cluster=-1, max_words=200):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
        if cluster_col is not None:
            df = df[df[cluster_col] == cluster]
        # text = ' '.join(content for content in df[constant.NEW_COL_LEMMA_TEXT])
        docs = [content for content in df[target_col]]
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
            max_words=max_words,
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

    def tf_idf(self, source_file_dir, source_filename, news_source=None, start_date=None, end_date=None,
               target_col=constant.NEW_COL_LEMMA_TEXT, cluster_col=None, cluster=-1, stance=None):
        """
        TF:

        IDF:

        """
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        if start_date is not None and end_date is not None:
            df = df[(start_date <= df['发布时间']) & (df['发布时间'] < end_date)]
        if cluster_col is not None:
            df = df[df[cluster_col] == cluster]
        if stance is not None:
            df = df[df[constant.COL_STANCE] == stance]
        docs = [content for content in df[target_col]]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=TFIDF_STOPWORDS)
        self.tfidf_X = self.tfidf_vectorizer.fit_transform(docs)
        data = {
            'word': self.tfidf_vectorizer.get_feature_names(),
            'tfidf': self.tfidf_X.toarray().sum(axis=0).tolist()
        }
        tfidf_df = pd.DataFrame(data).sort_values(by="tfidf", ascending=False, ignore_index=True)
        return self.tfidf_X, tfidf_df

    def tf_idf_by_date(self, source_file_dir, source_filename, news_source=None, start_date='2021-7-16 00:00:0', stop_date='2021-8-9 00:00:0'):
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

    def sentiment_analysis_by_textblob(self, source_file_dir, source_filename, dest_file_dir, dest_filename,
                                       ignore=None, col_name='语境'):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df['极性'] = 0
        df['主观性'] = 0
        df['立场'] = ''
        for idx, row in df.iterrows():
            context = row[col_name]
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
                              label='极性', cluster=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if cluster is not None:
            df = df[df[constant.NEW_COL_CONTENT_CLUSTER] == cluster]
        # sns.displot(df["立场"], height=5, aspect=1.8)
        # df.groupby('立场').count().plot(kind='pie', y='立场', autopct='%1.0f%%')
        plot.pie(x=df['立场'].value_counts(), labels=df['立场'].value_counts().index, autopct='%1.1f%%')
        plot.title(label)
        plot.savefig(os.path.join(dest_file_dir, dest_filename))

    def kmeans(self, source_file_dir, source_filename, target_col, k, cluster_col_name, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        # 使用TF-IDF向量化内容
        docs = [content for content in df[target_col]]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        # 使用K-means算法进行聚类分析
        kmeans = KMeans(n_clusters=k, random_state=112)
        kmeans.fit(X)
        # 获取每个新闻标题所属的类别
        labels = kmeans.labels_
        # 统计每个类别的数量
        cluster_counts = np.bincount(labels)
        # 使用主成分分析进行降维
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X.toarray())
        # 生成两子图
        fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(10, 5))
        # 根据聚类类别绘制不同颜色的散点图
        for i in range(k):
            ax1.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=f"群集{i + 1}")
        # 添加图例和标题
        ax1.legend()
        ax1.set_title("聚类散点图")
        # 绘制柱状图表示类别数量
        ax2.bar(range(1, k+1), cluster_counts)

        # 添加标签和标题
        ax2.set_xlabel('群集')
        ax2.set_ylabel('新闻数量')
        ax2.set_title('每个群集中新闻的数量')
        # 为每个条形图添加数值标签
        for x, y in enumerate(cluster_counts):
            plot.text(x + 1, y + 1, y, ha='center')
        # 输出每个新闻标题所属的类别
        # for i, title in enumerate(docs):
        #     label = kmeans.labels_[i]
        #     print(f"标题: {title}，类别: {label}")
        # 在excel中标注类别
        df[cluster_col_name] = -1
        for idx, row in df.iterrows():
            label = kmeans.labels_[idx]
            # print('idx:' + str(idx) + ', cluster: ' + str(label))
            df.loc[idx, cluster_col_name] = label
        # df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)
        # 显示图表
        plot.show()

    def context_of_keyword(self,  source_file_dir, source_filename, keyword, dest_file_dir, dest_filename,
                           target_col=constant.COL_CONTENT,
                           cluster_name=constant.NEW_COL_CONTENT_CLUSTER, cluster=2, gap=5):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df = df[df[cluster_name] == cluster]
        keyword = keyword.lower()
        keyword_size = len(keyword.split(' '))
        ids = []
        sent_indexes = []
        keywords = []
        toks = []
        poses = []
        positions = []
        for idx, row in df.iterrows():
            content = row[target_col].lower()
            id = row[constant.COL_ID]
            print('doc####', id)
            doc = self.nlp(content)
            for sent_index, sentence in enumerate(doc.sents):
                sent_txt = sentence.text
                if keyword in sent_txt:
                    special_colon = False
                    if sent_index == 0:
                        special_colon = True
                    tokens, sent_txt = self.split_sent_to_tokens(sentence,  ['19th', 'non'], special_colon)
                    indexes = self.indexes_of_tokens(keyword, tokens)
                    sent_after_toks = []
                    for index in indexes:
                        print('index of keyword in sentence-----' + str(index))
                        pre_i, suf_i = utils.i_range(index, keyword_size, gap)
                        for i, token in enumerate(tokens):
                            if pre_i <= i < index:
                                if token.lemma_.strip().lower() in sent_after_toks:
                                    continue
                                ids.append(id)
                                sent_indexes.append(sent_index)
                                keywords.append(keyword)
                                toks.append(token.lemma_.strip().lower())
                                poses.append(token.pos_)
                                positions.append('before')
                            if index + keyword_size - 1 < i < suf_i:
                                sent_after_toks.append(token.lemma_.strip().lower())
                                ids.append(id)
                                sent_indexes.append(sent_index)
                                keywords.append(keyword)
                                toks.append(token.lemma_.strip().lower())
                                poses.append(token.pos_)
                                positions.append('after')
                            if i == index:
                                sent_after_toks = []
        data = {
            constant.COL_ID: ids,
            constant.COL_SENT_INDEX: sent_indexes,
            constant.COL_KEYWORD: keywords,
            constant.COL_CONTEXT: toks,
            constant.COL_POS: poses,
            constant.COL_POSITION: positions,
        }
        context_df = pd.DataFrame(data)
        context_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def split_sent_to_tokens(self, sentence, skip_words=[], special_colon=False):
        tokens = []
        colon_index = -1
        if special_colon:
            for i, token in enumerate(sentence):
                if ':' == token.text:
                    colon_index = i
                    break
        for i, token in enumerate(sentence):
            if token.is_stop:
                continue
            if token.is_punct:
                continue
            if token.is_space:
                continue
            if token.is_digit:
                continue
            if token.text in skip_words:
                continue
            if i > colon_index:
                tokens.append(token)
        return tokens, ' '.join([token.text for token in tokens])

    def indexes_of_tokens(self, keyword, tokens):
        keywords = keyword.split(' ')
        indexes = []
        size = len(tokens)
        for i, token in enumerate(tokens):
            if token.text == keywords[0]:
                matched = True
                for j, key in enumerate(keywords):
                    if i + j < size:
                        if key != tokens[i + j].text:
                            matched = False
                            break
                if matched:
                    indexes.append(i)
        return indexes

    def word_pos_from_context_file(self, source_file_dir, source_filename, context_file_dir, context_filename,
                                   overwrite=False, top=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        context_df = self.excel_obj.read_excel(context_file_dir, context_filename)
        if overwrite:
            df[constant.COL_POS] = ''
        for idx, row in df.iterrows():
            if top is not None and idx > 200:
                break
            word = row[constant.COL_WORD]
            pos = row[constant.COL_POS]
            if pos == '' or pd.isna(pos):
                print('doc####', word)
                for idx2, row2 in context_df.iterrows():
                    context_word = row2[constant.COL_CONTEXT]
                    if word == context_word:
                        context_pos = row2[constant.COL_POS]
                        df.loc[idx, constant.COL_POS] = context_pos
                        break
        df.to_excel(os.path.join(source_file_dir, source_filename), index=False)

    def appraisal_attitude_analysis(self, source_file_dir, source_filename,
                                    appraisal_attitude_file_dir, appraisal_attitude_filename,
                                    target_file_dir, target_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        app_att_df = self.excel_obj.read_excel(appraisal_attitude_file_dir, appraisal_attitude_filename)
        # df[constant.COL_POS] = ''
        for idx, row in df.iterrows():
            # if idx > 200:
            #     break
            word = row[constant.COL_WORD]
            for idx2, row2 in app_att_df.iterrows():
                word2 = row2[constant.COL_WORD]
                if word == word2:
                    df.loc[idx, constant.COL_SUBSYSTEM] = row2[constant.COL_SUBSYSTEM]
                    df.loc[idx, constant.COL_TYPE] = row2[constant.COL_TYPE]
                    df.loc[idx, constant.COL_SUBTYPE] = row2[constant.COL_SUBTYPE]
                    df.loc[idx, constant.COL_EVALUATION] = row2[constant.COL_EVALUATION]
                    break
        df.to_excel(os.path.join(target_file_dir, target_filename), index=False)

    def lemma_text_to_sentences(self, source_file_dir, source_filename,
                                    dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        docs = {}
        doc_ids = []
        doc_titles = []
        doc_clusters = []
        doc_sents = []
        for idx, row in df.iterrows():
            content = row[constant.NEW_COL_LEMMA_TEXT]
            id = row[constant.COL_ID]
            title = row[constant.COL_TITLE]
            cluster = row[constant.NEW_COL_CONTENT_CLUSTER]
            lines = content.split('.')
            print('doc####', id)
            for i, line in enumerate(lines):
                line = line.strip()
                if line == '':
                    continue
                tmp = line.split(' ')
                if len(tmp) <= 5:
                    continue
                doc_ids.append(id)
                doc_titles.append(title)
                doc_clusters.append(cluster)
                doc_sents.append(line)
        docs[constant.COL_ID] = doc_ids
        docs[constant.COL_TITLE] = doc_titles
        docs[constant.NEW_COL_CONTENT_CLUSTER] = doc_clusters
        docs[constant.COL_SENTENCE] = doc_sents
        context_df = pd.DataFrame(docs, columns=[constant.COL_ID, constant.COL_TITLE,
                                                 constant.NEW_COL_CONTENT_CLUSTER, constant.COL_SENTENCE])
        context_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def plot_sentence_count_by_cluster(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df1 = df.groupby(constant.COL_TOPIC).count()
        plot.title('三大亚运主题语句数量分布')
        plot.xlabel('亚运主题')
        plot.ylabel('语句数量')
        plot.bar(['政治话题', '比赛战报', '杭州印象'], df1[constant.COL_ID])
        # 为每个条形图添加数值标签
        for x, y in enumerate(df1[constant.COL_ID]):
            plot.text(x, y + 2, y, ha='center')
        # plot.show()
        plot.savefig(os.path.join(dest_file_dir, dest_filename))

    def generate_keywords_by_pos(self, source_file_dir, source_filename, dest_file_dir, dest_filename, filter_pos):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        ids = []
        keywords = []
        poses = []
        frequencies = []
        types = []
        for idx, row in df.iterrows():
            print('doc#####' + str(idx))
            word_counts = {}
            content = row[constant.COL_CONTENT_EN]
            id = row[constant.COL_ID]
            type = row[constant.COL_TYPE]
            doc = self.nlp(content)
            if filter_pos == 'NOUN_TRUNKS':
                for chunk in doc.noun_chunks:
                    txt = chunk.text
                    count = word_counts.get(txt, 0)
                    count += 1
                    word_counts[txt] = count
            else:
                for token in doc:
                    pos = token.pos_
                    if pos == filter_pos:
                        txt = token.text
                        count = word_counts.get(txt, 0)
                        count += 1
                        word_counts[txt] = count
            for word in word_counts:
                ids.append(id)
                types.append(type)
                keywords.append(word)
                poses.append(filter_pos)
                frequencies.append(word_counts[word])
        data = {
            constant.COL_ID: ids,
            constant.COL_KEYWORD: keywords,
            constant.COL_POS: poses,
            constant.COL_FREQUENCY: frequencies,
            constant.COL_TYPE: types,
        }
        propn_df = pd.DataFrame(data)
        propn_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def generate_keywords_by_pos_by_type(self, source_file_dir, source_filename, dest_file_dir, dest_filename, filter_pos, type):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df = df[df[constant.COL_TYPE] == type]
        keywords = []
        poses = []
        frequencies = []
        types = []
        word_counts = {}
        for idx, row in df.iterrows():
            print('doc#####' + str(idx))
            content = row[constant.COL_CONTENT_EN]
            doc = self.nlp(content)
            if filter_pos == 'NOUN_TRUNKS':
                for chunk in doc.noun_chunks:
                    txt = chunk.text
                    count = word_counts.get(txt, 0)
                    count += 1
                    word_counts[txt] = count
            else:
                for token in doc:
                    pos = token.pos_
                    if pos == filter_pos:
                        txt = token.text
                        count = word_counts.get(txt, 0)
                        count += 1
                        word_counts[txt] = count
        for word in word_counts:
            types.append(type)
            keywords.append(word)
            poses.append(filter_pos)
            frequencies.append(word_counts[word])
        data = {
            constant.COL_KEYWORD: keywords,
            constant.COL_POS: poses,
            constant.COL_FREQUENCY: frequencies,
            constant.COL_TYPE: types,
        }
        propn_df = pd.DataFrame(data)
        propn_df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

# word class: ADJ, NOUN, PROPN, ADV, VERB
# word_class = 'NOUN'
# DataAnalysis().generate_word_cloud_of_sentiment(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, constant.DIR_SENTIMENT),
#                                                 'ceremony_2021-7-22_2021-7-24_textblob_sentiment.xlsx',
#                                                 os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED,
#                                                              constant.DIR_WORDCLOUD, 'ceremony_'+word_class.lower()+'.png'),
#                                                 None, more_stopwords=None, word_class=word_class)
# print("-----------------------------------------------")
# sentiment_wc_stopwords = ['japanese', 'australian', 'israeli', 'palestinian']
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
