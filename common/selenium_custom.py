from selenium import webdriver


def use_opened_chrome():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    chrome_driver = "/usr/local/bin/chromedriver"
    driver = webdriver.Chrome(executable_path=chrome_driver, options=options)
    return driver


def use_opened_chrome_windows():
    # cd 'C:/Program Files/Google/Chrome/Application'
    # ./chrome.exe --remote-debugging-port=9527
    options = webdriver.ChromeOptions()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9527")
    options.add_argument('--max-memory-percent=100')
    driver = webdriver.Chrome(options=options)
    return driver

def safe_find_elm(elm, xpath,):
    try:
        found_elm = elm.find_element_by_xpath(xpath)
        return found_elm
    except:
        return None


def safe_find_elms(elm, xpath):
    try:
        found_elms = elm.find_elements_by_xpath(xpath)
        return found_elms
    except:
        return None


def safe_find_elms_tag(elm, tag_name):
    try:
        found_elm = elm.find_elements_by_tag_name(tag_name)
        return found_elm
    except:
        return None