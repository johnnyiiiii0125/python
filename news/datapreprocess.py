import os
from excel import Excel
import nltk
import spacy
from common import constant
from spacy.lang.en.stop_words import STOP_WORDS

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
        df[constant.NEW_COL_NOISE_FLAG] = 0
        for idx, row in df.iterrows():
            content = row['内容'].lower()
            title = row['标题'].lower()
            if 'olympic' not in content or 'tokyo' not in content:
                df.loc[idx, constant.NEW_COL_NOISE_FLAG] = 1
            # 经过上一步过滤，还是有很多无用数据，因此对标题进行筛选
            if 'olympic' not in title and 'tokyo' not in title \
                    and 'japan' not in title and 'japanese' not in title and 'olympian' not in title:
                df.loc[idx, constant.NEW_COL_NOISE_FLAG] = 1
        df.to_excel(os.path.join(dest_file_dir, dest_filename), index=False)

    def remove_content_linebreak(self, source_file_dir, source_filename, dest_file_dir, dest_filename):
        df = self.excel_obj.read_excel(source_file_dir, source_filename)
        df[constant.NEW_COL_CONTENT_CLEAN] = ''
        for idx, row in df.iterrows():
            content = row['内容'].lower()
            ctn = content.replace('\n', ' ').replace('\r', ' ')
            df.loc[idx, constant.NEW_COL_CONTENT_CLEAN] = ctn
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
        new_col = constant.NEW_COL_LEMMA_TEXT
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
