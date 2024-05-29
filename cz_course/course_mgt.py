import tkinter as tk
from selenium import webdriver
import threading
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd

URL = 'http://es.zjczxy.cn/'


class CourseMgt:

    def __init__(self):
        self.driver = None
        self.root = None
        self.is_opened = False
        self.msg = None

    def open_browser(self):
        if self.driver:
            return
        self.driver = webdriver.Chrome()
        #self.driver = selenium_custom.use_opened_chrome_windows()
        threading.Thread(target=self.driver.get, args=(URL,), daemon=True)

    def open_jw(self):
        self.driver.get(URL)

    def check_input_page(self):
        all_handles = self.driver.window_handles
        self.is_opened = False
        for handle in all_handles:
            self.driver.switch_to.window(handle)
            window_url = self.driver.current_url
            if 'XSCJ' in window_url:
                self.is_opened = True
                break
        if self.is_opened:
            messagebox.showinfo(title='检测结果', message='成绩录入页已确认！')
        else:
            messagebox.showinfo(title='检测结果', message='未检测到成绩录入页！')

    def select_file(self):
        if not self.is_opened:
            messagebox.showinfo(title='错误', message='请先打开成绩录入页！')
            return
        filename = filedialog.askopenfilename()
        if filename == '':
            messagebox.showinfo(title='选择结果', message='未选中文件！')
            return
        messagebox.showinfo(title='选择结果', message=filename)
        self.fill_courses(filename)

    def fill_courses(self, filename):
        df = pd.read_excel(filename)
        iframe = self.driver.find_element_by_xpath('//iframe[@name="frmRpt"]')
        self.driver.switch_to.frame(iframe)
        form1 = self.driver.find_element_by_xpath('//form[@name="form1"]')
        tables = form1.find_elements_by_tag_name('table')
        main_table = tables[len(tables)-1]
        trs = main_table.find_elements_by_tag_name('tr')
        for tr in trs:
            tr_id = tr.get_attribute('id')
            if 'hh' in tr_id:
                tr_index = tr_id.replace('hh', '')
                #threading.Thread(target=self.refresh_msg, args=('正在录入第' + tr_index + '个学生。。。',), daemon=True)
                td_xuehao = tr.find_element_by_id('tr' + tr_index + '_yhxh')
                td_pscj = tr.find_element_by_id('tr' + tr_index + '_pscj')
                pscj_input = td_pscj.find_element_by_tag_name('input')
                td_qmcj = tr.find_element_by_id('tr' + tr_index + '_qmcj')
                qmcj_input = td_qmcj.find_element_by_tag_name('input')
                for idx, row in df.iterrows():
                    xuehao = str(row['学号']).strip()
                    pingshi = row['平时成绩']
                    qimo = row['期末成绩']
                    if xuehao == td_xuehao.text:
                        pscj_input.send_keys(str(pingshi))
                        qmcj_input.send_keys(str(qimo))
                        break
        self.msg.config(text='完成录入！！')

    def refresh_msg(self, content):
        self.msg.config(text=content)

    def initWindow(self):
        # 创建用户界面
        self.root = tk.Tk()
        self.root.title('成绩填充系统')
        self.root.geometry('500x500')
        #website_msg = tk.Label(self.root, text='网址：http://es.zjczxy.cn/')
        #website_msg.pack()
        btn_open_browser = tk.Button(self.root, text='打开浏览器', command=self.open_browser)
        btn_open_browser.pack()
        btn_open_jw = tk.Button(self.root, text='打开教务系统', command=self.open_jw)
        btn_open_jw.pack()
        btn_check_input_page = tk.Button(self.root, text='检测并切换成绩录入页', command=self.check_input_page)
        btn_check_input_page.pack()
        btn_select_file = tk.Button(self.root, text='选择excel', command=self.select_file)
        btn_select_file.pack()
        self.msg = tk.Label(self.root, text='')
        self.msg.pack()
        # 启动主循环
        self.root.mainloop()


cm = CourseMgt()
cm.initWindow()
