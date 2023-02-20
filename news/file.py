import logging
import os
import json
import shutil

logger = logging.getLogger('news_query')
logging.basicConfig(level=logging.INFO)  # 设置日志级别


def write_to_file(dir, file_name, content):
    write_to_file(dir, file_name, content, 'w')


def write_to_file(dir, file_name, content, mode):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fo = open(os.path.join(dir, file_name), mode)
    fo.write(content)
    fo.close()


def list_dirs(dir):
    files = os.listdir(dir)
    return files


def read_file_to_json(dir, file_name):
    text = read_file_content(dir, file_name)
    data = json.loads(text)
    return data


def write_lines_to_file(dir, file_name, lines):
    write_lines_to_file(dir, file_name, lines, 'w')


def write_lines_to_file(dir, file_name, lines, mode):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fo = open(os.path.join(dir, file_name), mode)
    for line in lines:
        fo.write(line + '\n')
    fo.close()


def read_file_content(dir, file_name):
    if dir:
        path = os.path.join(dir, file_name)
    else:
        path = file_name
    fo = open(path, 'r')
    text = fo.read()
    fo.close()
    return text


def read_file_lines(dir, file_name):
    if dir:
        path = os.path.join(dir, file_name)
    else:
        path = file_name
    fo = open(path, 'r')
    lines = fo.readlines()
    fo.close()
    return lines


def mv_file(source_dir, source_file, dest_dir, dest_file):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    try:
        shutil.move(os.path.join(source_dir, source_file), os.path.join(dest_dir, dest_file))
    except FileNotFoundError:
        print(os.path.join(source_dir, source_file) + ' not found!')


def remove_redundant_urls_in_file(urls, file_path):
    if not os.path.exists(file_path):
        return urls
    ret_urls = []
    file_lines = read_file_lines(None, file_path)
    for url in urls:
        if url+'\n' in file_lines:
            continue
        ret_urls.append(url)
    return ret_urls


def check_and_make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def dict_to_json_file(dict, dest_dir, dest_file):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    with open(os.path.join(dest_dir, dest_file), "w") as outfile:
        json.dump(dict, outfile)