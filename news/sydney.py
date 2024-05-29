from dataprocess import DataAnalysis
import os
from common import constant


class Sydney:
    def __init__(self):
        self.ROOT_DIR = 'D:\BaiduSyncdisk\Sydney\项目'
        self.data_filename = 'data.xlsx'

    def get_word_count_by_pos(self):
        DataAnalysis().generate_keywords_by_pos(source_file_dir=os.path.join(self.ROOT_DIR),
                                                source_filename=self.data_filename,
                                                dest_file_dir=os.path.join(self.ROOT_DIR), dest_filename='propn.xlsx',
                                                filter_pos='PROPN')

    def get_noun_trunks_count(self):
        DataAnalysis().generate_keywords_by_pos(source_file_dir=os.path.join(self.ROOT_DIR),
                                                source_filename=self.data_filename,
                                                dest_file_dir=os.path.join(self.ROOT_DIR), dest_filename='noun_trunks.xlsx',
                                                filter_pos='NOUN_TRUNKS')
    def get_word_count_by_pos_by_type(self):
        type = 2
        DataAnalysis().generate_keywords_by_pos_by_type(source_file_dir=os.path.join(self.ROOT_DIR),
                                                source_filename=self.data_filename,
                                                dest_file_dir=os.path.join(self.ROOT_DIR),
                                                        dest_filename='propn_type'+str(type)+'.xlsx',
                                                filter_pos='PROPN', type=type)

    def get_noun_trunks_count_by_type(self):
        type = 2
        DataAnalysis().generate_keywords_by_pos_by_type(source_file_dir=os.path.join(self.ROOT_DIR),
                                                source_filename=self.data_filename,
                                                dest_file_dir=os.path.join(self.ROOT_DIR),
                                                        dest_filename='noun_trunks_type'+str(type)+'.xlsx',
                                                filter_pos='NOUN_TRUNKS', type=type)

# Sydney().get_word_count_by_pos()
# Sydney().get_noun_trunks_count()
Sydney().get_word_count_by_pos_by_type()
Sydney().get_noun_trunks_count_by_type()