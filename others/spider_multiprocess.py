import pickle
import sys
import time
import random
from func_timeout import func_set_timeout, FunctionTimedOut
import requests
import json
from bs4 import BeautifulSoup
from multiprocessing import Process, Pool, Manager
import os

has_name = set()

def funtime(func):
    '''
    This decarator is used for timing
    :param func: the funtion you want to time.
    :return:
    '''
    def wrapper(*args, **kw):
        local_time = time.time()
        tmp = func(*args, **kw)
        print("current Function [%s] run time is %.6f" % (func.__name__, (time.time() - local_time)))
        return tmp

    return wrapper


class Spider:

    def __init__(self, un_spi, has_spi):

        self.base_url = 'https://www.zhihu.com/search?type=content&q='
        self.un_spi = un_spi
        self.has_spi = has_spi
    # @funtime
    def parse_one_page(self, response):
        '''
        parse the response and store the infomation into the .txt
        :param response: the http response
        :return:
        '''
        short_name = ''
        full_name = ''
        link_url = ''

        soup = BeautifulSoup(response.text, 'html.parser')


        # find competion products
        div = soup.find('div', attrs={'class':"RichContent-inner"})
        links = div.find('b').text
        sp = div.find('span').text
        a = random.randint(6,30)
        name = sp[:a]
        global has_name
        if name is not None and name not in has_name and not self.un_spi.full():
            self.un_spi.put(name)
            has_name.add(name)

        a = random.randint(2, 40)
        name = sp[:a]
        if name is not None and name not in has_name and not self.un_spi.full():
            self.un_spi.put(name)
            has_name.add(name)

        with open('data/mcompany.txt', 'a+', encoding='utf-8') as f:
            sp = sp + '\n'
            f.write(sp)

    # this decorator is usful because it can raise an exception when the function overran
    # @func_set_timeout(2)
    # @funtime
    def spider_name(self, name):
        '''

        :param name: the name of the person to be spidered
        :return:
        '''


        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
                'Cookie': 'ABTEST=3|1564648758|v17; IPLOC=JP; SUID=677BC7342113940A000000005D42A536; SUV=1564648795913366; browerV=3; osV=1'
            }
            response = requests.get(self.base_url+name, headers=headers, timeout=5)
            self.parse_one_page(response)
            self.has_spi.put(name)
            print("pid: %d put: %s, unspidered: %d" %(os.getpid(), name, self.un_spi.qsize()))
        except Exception as e:
            print(e)

    # @funtime
    def run_spider_name(self, name):
        '''
        this function mains to catch the FunctionTimedOut Exception raised by func spider_name()
        :param name:
        :return:
        '''
        try:
            self.spider_name(name)
        except FunctionTimedOut:
            # self.dumpToPkl()
            # print("dump done")
            sys.exit(0)




def spider(un_spidered, has_spidered):

    print("spider(%s),父进程为(%s)" % (os.getpid(), os.getppid()))
    a = Spider(un_spidered, has_spidered)
    while not un_spidered.empty():
        name = a.un_spi.get()
        a.run_spider_name(name)

if __name__ == '__main__':
    try:
        un_spidered = Manager().Queue(maxsize=100)
        has_spidered = Manager().Queue(maxsize=1000)

        load = False
        if load:
            with open('data/mcompany_spidered.pkl', 'wb') as f:
                pickle.dump(un_spidered, f)
            with open('data/mcompany_unspider.pkl', 'wb') as f:
                pickle.dump(un_spidered, f)
        else:
            un_spidered.put('陈乔恩')
            un_spidered.put('李宇春')
            un_spidered.put('贾乃亮')
            un_spidered.put('当队列空时，是一直')
            un_spidered.put('队列空否')
            print(un_spidered.qsize())

        po = Pool()
        for i in range(4):
            po.apply_async(func=spider, args=(un_spidered, has_spidered,))
        po.close()
        po.join()
    except Exception as e:
        print(e)
    finally:
        with open('data/mcompany_spidered.pkl', 'wb') as f:
            pickle.dump(has_spidered, f)
        with open('data/mcompany_unspider.pkl', 'wb') as f:
            pickle.dump(un_spidered, f)
