import re
import requests
import traceback
from urllib.parse import quote
import sys
import logging

class Crawler:
    '''爬百度搜索结果的爬虫'''
    url = ''
    urls = []
    o_urls = []
    html = ''
    total_pages = 5
    current_page = 0
    next_page_url = ''
    timeout = 60  # 默认超时时间为60秒
    all_o_urls = []  # 存储所有page爬取到的原始url
    all_urls = []  # 存储所有page爬取到的真实url
    headers_parameters = {  # 发送HTTP请求时的HEAD信息，用于伪装为浏览器
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Is_referer': "https://www.baidu.com/",
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0'
    }

    def __init__(self, keyword):
        self.url = f'https://www.baidu.com/baidu?wd={quote(keyword)}&tn=monline_dg&ie=utf-8'

    def set_timeout(self, time):
        '''设置超时时间，单位：秒'''
        try:
            self.timeout = int(time)
        except:
            pass

    def set_total_pages(self, num):
        '''设置总共要爬取的页数'''
        try:
            self.total_pages = int(num)
        except:
            pass

    def set_current_url(self, url):
        '''设置当前url'''
        self.url = url

    def switch_url(self):
        '''切换当前url为下一页的url
           若下一页为空，则退出程序'''
        if self.next_page_url == '':
            sys.exit()
        else:
            self.set_current_url(self.next_page_url)

    def is_finish(self):
        '''判断是否爬取完毕'''
        return self.current_page >= self.total_pages

    def get_html(self):
        '''爬取当前url所指页面的内容，保存到html中'''
        try:
            r = requests.get(self.url, timeout=self.timeout, headers=self.headers_parameters)
            if r.status_code == 200:
                self.html = r.text
                self.current_page += 1
            else:
                self.html = ''
                print('[ERROR]', self.url, 'get此url返回的http状态码不是200')
        except requests.RequestException as e:
            self.html = ''
            print('[ERROR]', self.url, '请求失败:', str(e))

    def get_urls(self):
        '''从当前html中解析出搜索结果的url，保存到o_urls'''
        self.o_urls = list(set(re.findall(r'href="(http://www\.baidu\.com/link\?url=.*?)"', self.html)))
        self.all_o_urls.extend(self.o_urls)
        self.next_page_url = f'https://www.baidu.com/s?wd={quote(keyword)}&pn={self.current_page * 10}'

    def get_real(self, o_url):
        '''获取重定向url指向的网址'''
        try:
            r = requests.get(o_url, allow_redirects=False)
            if r.status_code == 302:
                return r.headers.get('Location', o_url)
        except requests.RequestException:
            pass
        return o_url

    def transformation(self):
        '''读取当前o_urls中的链接重定向的网址，并保存到urls中'''
        self.urls = [self.get_real(o_url) for o_url in self.o_urls]
        self.all_urls.extend(self.urls)

    def save_all_urls(self, path: str = 'urls.txt'):
        '''输出当前urls中的url'''
        with open(path, "a") as f:
            for url in self.all_urls:
                f.write(url + "\n")

    # def print_urls(self):
    #     '''输出当前urls中的url'''
    #     for url in self.urls:
    #         print(url)

    # def print_o_urls(self):
    #     '''输出当前o_urls中的url'''
    #     for url in self.o_urls:
    #         print(url)

    def get_top_k_urls(self, k):
        '''获取前k个url'''
        return self.all_urls[:k]

    def run(self, save_path: str = None):
        while not self.is_finish():
            self.get_html()
            self.get_urls()
            self.transformation()
            self.switch_url()
        if save_path:
            self.save_all_urls(save_path)
        logging.info(f"Finished crawling {self.total_pages} pages of search results for keyword '{keyword}'!")

"""Baidu Search tool spec."""

import urllib.parse
from typing import Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.readers.web import BeautifulSoupWebReader

class BaiduSearchToolSpec(BaseToolSpec):
    """Baidu Search tool spec."""

    spec_functions = ["baidu_search"]

    def __init__(self, timeout: int = 60, num_pages: int = 1, top_k: int = 1) -> None:
        """Initialize with parameters."""
        self.timeout = timeout
        self.num_pages = num_pages
        self.top_k = top_k

    def baidu_search(self, query: str):
        """
        Make a query to the Baidu search engine to receive a list of results.

        args:
            query (str): The query to be passed to Baidu search.

        """
        crawler = Crawler(keyword=query)
        crawler.set_timeout(self.timeout)
        crawler.set_total_pages(self.num_pages)
        crawler.run()
        top_urls = crawler.get_top_k_urls(self.top_k)
        assert isinstance(top_urls, list)
        reader = BeautifulSoupWebReader()
        return reader.load_data(top_urls)


# functional version of BaiduSearchToolSpec
def baidu_search(query: str, timeout: int = 60, num_pages: int = 1, top_k: int = 1):
    """
    Make a query to the Baidu search engine to receive a list of results.

    args:
        query (str): The query to be passed to Baidu search.

    """
    crawler = Crawler(keyword=query)
    crawler.set_timeout(timeout)
    crawler.set_total_pages(num_pages)
    crawler.run()
    top_urls = crawler.get_top_k_urls(top_k)
    assert isinstance(top_urls, list)
    # reader = BeautifulSoupWebReader()
    # result = reader.load_data(top_urls)
    doc_ls = []
    for url in top_urls:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                doc_ls.append(Document(text=r.text))
            else:
                print('[ERROR]', url, 'get此url返回的http状态码不是200')
        except requests.RequestException as e:
            print('[ERROR]', url, '请求失败:', str(e))
    return doc_ls

if __name__ == '__main__':
    # 示例用法
    keyword = 'INFP'
    c = Crawler(keyword)
    c.set_total_pages(2)
    c.run(save_path=keyword+'.txt')
    top_urls = c.get_top_k_urls(10)
    print("Top 10 URLs:")
    for url in top_urls:
        print(url)
    
    # test BaiduSearchToolSpec
    baidu_spec = BaiduSearchToolSpec()
    results = baidu_spec.baidu_search(keyword)
    print("Finished loading search results!")

