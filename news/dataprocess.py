import os
import db
import file
import pandas as pd
from common import constant
import nltk
import spacy
import matplotlib.pyplot as plot
from wordcloud import WordCloud
# from gensim import corpora, models
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation

NEW_COL_LEMMA_TEXT = '内容lemma后纯文本'
NEW_COL_NOISE_FLAG = '噪音行'
NEW_COL_CONTENT_CLEAN = '内容整行'
COL_ID = 'id'


# 2021年7月23日开幕、8月8日闭幕

class Excel:
    def __init__(self):
        self.db_obj = db.DB()
        self.header = ['id', '媒体', '新闻链接', '标题', '内容', '发布时间', '添加时间', '修改时间']

    def data_from_db_to_excel(self, sql, file_dir, file_name):
        result = self.db_obj.query_results(sql)
        data = result['data']
        if len(data) > 0:
            file.check_and_make_dirs(file_dir)
            df = pd.DataFrame(list(data), columns=self.header)
            df.to_excel(os.path.join(file_dir, file_name), index=False)
        self.db_obj.close_conn()

    def read_excel(self, file_dir, filename):
        df = pd.read_excel(os.path.join(file_dir, filename))
        return df


class DataPreprocess:
    def __init__(self):
        self.excel_obj = Excel()
        self.nlp = spacy.load('en_core_web_lg')
        # add more stop words--------not working
        STOP_WORDS.add('say')
        STOP_WORDS.add('year')
        STOP_WORDS.add('said')
        self.pos_list = ['NOUN', 'PROPN', 'ADJ', 'VERB']

    # 1st step: remove null/empty content rows
    def remove_null_rows(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df.dropna(axis='index', subset=['内容'], inplace=True)  # 去除内容列为空的行（共39行）
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def keep_news_in_2021(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df.drop(df[~(df['发布时间'].map(lambda d: d.year)).isin([2021])].index, inplace=True)
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def keep_news_in_date_range(self, source_file_dir, source_filename, start_time, end_time,
                                dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df.drop(df[(df['发布时间'] < start_time) | (df['发布时间'] > end_time)].index, inplace=True)
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def remove_noise(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df[NEW_COL_NOISE_FLAG] = 0
        for idx, row in df.iterrows():
            content = row['内容'].lower()
            title = row['标题'].lower()
            if 'olympic' not in content or 'tokyo' not in content:
                df.loc[idx, NEW_COL_NOISE_FLAG] = 1
            # 经过上一步过滤，还是有很多无用数据，因此对标题进行筛选
            if 'olympic' not in title and 'tokyo' not in title \
                    and 'japan' not in title and 'japanese' not in title and 'olympian' not in title:
                df.loc[idx, NEW_COL_NOISE_FLAG] = 1
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def remove_content_linebreak(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df[NEW_COL_CONTENT_CLEAN] = ''
        for idx, row in df.iterrows():
            content = row['内容'].lower()
            ctn = content.replace('\n', ' ').replace('\r', ' ')
            df.loc[idx, NEW_COL_CONTENT_CLEAN] = ctn
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def tokenization_nltk(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        new_col = '内容分词'
        df[new_col] = ''
        for idx, row in df.iterrows():
            content = row['内容']
            tokens = nltk.tokenize.word_tokenize(content)
            df.loc[idx, new_col] = ' '.join(tokens)
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def token_stopwords_lemma_spacy(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        new_col = NEW_COL_LEMMA_TEXT
        df[new_col] = ''
        for idx, row in df.iterrows():
            print('doc#####' + str(idx))
            content = row['内容']
            doc = self.nlp(content)
            doc_remain = []
            for token in doc:
                if not token.is_stop:
                    # 非stop words，
                    if not token.is_punct:
                        # remove punctuation
                        txt = token.lemma_.strip().lower()
                        # make sure the lemma is not a stop word
                        if txt not in STOP_WORDS:
                            if txt != '':
                                doc_remain.append(txt)
            df.loc[idx, new_col] = ' '.join(doc_remain)
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def generate_noun_chunks(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        new_col = '名词短语'
        df[new_col] = ''
        for idx, row in df.iterrows():
            print('doc#####' + str(idx))
            content = row['内容']
            doc = self.nlp(content)
            chunks = []
            for chunk in doc.noun_chunks:
                chunks.append(chunk.text)
            df.loc[idx, new_col] = ','.join(chunks)
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def filter_word_counts_by_frequency(self, word_counts, threshold):
        word_counts_frequency = {}
        pos_list = word_counts.keys()
        for pos in pos_list:
            word_counts_frequency[pos] = {}
            for word in word_counts[pos]:
                if word_counts[pos][word] >= threshold:
                    word_counts_frequency[pos][word] = word_counts[pos][word]
        return word_counts_frequency

    def word_counts_by_pos(self, text, pos_list, word_counts=None, threshold=None):
        doc = self.nlp(text)
        if word_counts is None:
            word_counts = {}
        for pos in pos_list:
            if pos not in word_counts:
                word_counts[pos] = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                pos = token.pos_
                if pos in pos_list:
                    lemma = token.lemma_.strip().lower()
                    # make sure the lemma is also not a stop word
                    if lemma in STOP_WORDS:
                        continue
                    count = word_counts[pos].get(lemma, 0)
                    count += 1
                    word_counts[pos][lemma] = count
        if threshold is not None:
            # filter the word counts dic
            word_counts = self.filter_word_counts_by_frequency(word_counts, threshold)
        return word_counts

    def word_counts_all_dataset(self, source_file_dir, source_filename, by_news_source=False):
        """
            if by_news_source = False:
            {
                'doc_id':{
                    'POS_TYPE':{
                        'word': count
                        ...
                    }
                    ...
                }
                ...
            }
            if by_news_source = True:
            {
                'news_source':{
                    'POS_TYPE':{
                        'word': count
                        ...
                    }
                    ...
                }
                ...
            }
        """
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        word_counts_all = {}
        for idx, row in df.iterrows():
            content = row['内容']
            id = row['id']
            news_source = row['媒体']
            print('doc#####', str(idx), '-----', id, '-----', news_source)
            words_count = self.word_counts_by_pos(content, self.pos_list, threshold=None)
            if by_news_source:
                if news_source not in word_counts_all:
                    word_counts_all[news_source] = {}
                for pos in words_count:
                    if pos not in word_counts_all[news_source]:
                        word_counts_all[news_source][pos] = {}
                    words_count_pos = words_count[pos]
                    for word in words_count_pos:
                        if word not in word_counts_all[news_source][pos]:
                            word_counts_all[news_source][pos][word] = words_count_pos[word]
                        else:
                            word_counts_all[news_source][pos][word] = word_counts_all[news_source][pos][word] + \
                                                                      words_count_pos[word]
            else:
                word_counts_all[id] = words_count
        return word_counts_all

    def sort_word_counts(self, word_counts_all, top, by_news_source=False):
        if by_news_source:
            # sort in news_source
            for news_source in word_counts_all:
                sorted_in_pos = word_counts_all[news_source]
                for pos in sorted_in_pos:
                    word_counts_pos = sorted_in_pos[pos]
                    sorted_in_pos[pos] = sorted(word_counts_pos.items(), key=lambda x: x[1], reverse=True)[:top]
            return word_counts_all
        else:
            sorted_in_pos = {}
            # add up all
            for id in word_counts_all:
                doc_word_counts = word_counts_all[id]
                print('word_counts', id)
                for pos in doc_word_counts:
                    if pos not in sorted_in_pos:
                        sorted_in_pos[pos] = {}
                    word_counts_in_pos = doc_word_counts[pos]
                    for word in word_counts_in_pos:
                        if word not in sorted_in_pos[pos]:
                            sorted_in_pos[pos][word] = word_counts_in_pos[word]
                        else:
                            sorted_in_pos[pos][word] = sorted_in_pos[pos][word] + word_counts_in_pos[word]
            # sort in pos
            for pos in sorted_in_pos:
                word_counts_pos = sorted_in_pos[pos]
                sorted_in_pos[pos] = sorted(word_counts_pos.items(), key=lambda x: x[1], reverse=True)[:top]
            return sorted_in_pos


class DataAnalysis:
    def __init__(self):
        self.excel_obj = Excel()
        self.tfidf_vectorizer = None
        self.tfidf_X = None
        self.lda = None

    def generate_word_cloud(self, source_file_dir, source_filename, saved_file_path, news_source=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        text = ' '.join(content for content in df[NEW_COL_LEMMA_TEXT])
        # Generate word cloud
        word_cloud = WordCloud(
            width=3000,
            height=2000,
            random_state=1,
            background_color="white",
            # colormap="Pastel1",
            collocations=False,
        ).generate(text)
        plot.imshow(word_cloud)
        plot.axis('off')
        # plot.show()
        word_cloud.to_file(saved_file_path)

    def total_word_count(self, source_file_dir, source_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        all_unique_words = []
        for idx, row in df.iterrows():
            content = row[NEW_COL_LEMMA_TEXT]
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

    def tf_idf(self, source_file_dir, source_filename, news_source=None):
        """
        TF:

        IDF:

        """
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        if news_source is not None:
            df = df[df['媒体'] == news_source]
        docs = [content for content in df[NEW_COL_LEMMA_TEXT]]
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_X = self.tfidf_vectorizer.fit_transform(docs)
        data = {
            'word': self.tfidf_vectorizer.get_feature_names(),
            'tfidf': self.tfidf_X.toarray().sum(axis=0).tolist()
        }
        tfidf_df = pd.DataFrame(data).sort_values(by="tfidf", ascending=False, ignore_index=True)
        return self.tfidf_X, tfidf_df

    def clustering_kmeans(self, data, k):
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=20000, random_state=0)
        y_means = kmeans.fit_predict(data)
        print(y_means)

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

    def context_by_keyword(self, source_file_dir, source_filename, keyword, context_length=10,
                           news_source=None, allocation=None):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        dest_filename = keyword + '_all.csv'
        if news_source is not None:
            df = df[df['媒体'] == news_source]
            if allocation is None:
                dest_filename = keyword + '_' + news_source + '.csv'
            else:
                dest_filename = keyword + '_' + "_".join(allocation) + "_" + news_source + '.csv'
        docs_keyword_contexts = {}
        for idx, row in df.iterrows():
            content = row[NEW_COL_LEMMA_TEXT]
            id = row[COL_ID]
            tokens = content.split(' ')
            # print('doc####', id)
            doc_keyword_contexts = []
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
                        doc_keyword_contexts.append(keyword_context.strip())
                    else:
                        all_allocation_is_in = True
                        for alloc in allocation:
                            if alloc not in keyword_context.strip():
                                all_allocation_is_in = False
                                break
                        if all_allocation_is_in:
                            doc_keyword_contexts.append(keyword_context.strip())
            if len(doc_keyword_contexts) > 0:
                docs_keyword_contexts[id] = doc_keyword_contexts
        lines = []
        lines.append('文档序号,语境')
        for k, v in docs_keyword_contexts.items():
            lines.append(str(k) + ',' + ','.join(v))
        file.write_lines_to_file(source_file_dir, dest_filename, lines, 'w')


# contexts
# DataAnalysis().context_by_keyword(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                    'news_token_lemma.xlsx', 'covid', 5,
#                                   news_source=constant.SOURCE_CNN, allocation=['case'])
# tf idf  && kmeans
# dataAnalysis = DataAnalysis()
# X, tfidf_df = dataAnalysis.tf_idf(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                   'news_token_lemma.xlsx', news_source=constant.SOURCE_CNN)
# tfidf_df.to_excel(os.path.join(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
#                                'tfidf_' + constant.SOURCE_BBC + '.xlsx'), index=True)
# lda = dataAnalysis.lda_model(2)
# dataAnalysis.print_lda_top_words(lda, dataAnalysis.tfidf_vectorizer.get_feature_names(), 20)
# DataAnalysis().clustering_kmeans(X, 3)

# print(DataPreprocess().nlp.Defaults.stop_words)

# DataAnalysis().lda_model(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx', news_source='cnn')

# get all words count: 34621
# DataAnalysis().total_word_count(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx')

# generate word cloud
# DataAnalysis().generate_word_cloud(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED), 'news_token_lemma.xlsx',
#             os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED, 'wc_all.png')
#                                    # , news_source='abc'
#                                    )

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
DataPreprocess().keep_news_in_date_range(os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
                                         'news_token_lemma.xlsx',
                                         '2021-07-16 00:00:0', '2021-08-09 00:00:0',
                                         os.path.join(constant.ROOT_DIR, constant.DIR_PROCESSED),
                                         'news_token_lemma-0716-0808.xlsx')

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
