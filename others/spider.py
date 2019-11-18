import pickle
import sys
import time
import random
from func_timeout import func_set_timeout, FunctionTimedOut
import requests
import json
from bs4 import BeautifulSoup

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

    def __init__(self, init=True):

        self.company_spidered = set()
        self.company_unspider = set()
        if init:
            self.company_unspider.add('https://www.tianyancha.com/brand/b9b6d92307')
        else:
            self.loadFromPkl()

    @funtime
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

        # find the name of this company

        try:
            link_url = response.url
            print(link_url)

            box = soup.find('div', attrs={'id':'project_web_top'})

            short_name = box.find('div', attrs={'class':'name'}).text
            full_name = box.find('div', attrs={'class':'infos'}).find('a',attrs={'class':'link-click'}).text

            print(short_name, full_name, link_url)
        except Exception as e:
            print(e)


        # find competion products
        div = soup.find('div', attrs={'id':"_container_jinpin"})
        links = div.find_all('a', attrs={'class':'link-click'})
        for link in links:
            url = link.get('href')
            if url is not None and url not in a.company_spidered:
                a.company_unspider.add(url)

        with open('data/company.txt', 'a+', encoding='utf-8') as f:
            if short_name is not None and full_name is not None:
                line = short_name + '\t' + full_name + '\t' + link_url + '\n'
                f.write(line)

    # this decorator is usful because it can raise an exception when the function overran
    @func_set_timeout(2)
    @funtime
    def spider_name(self, url_name):
        '''

        :param name: the name of the person to be spidered
        :return:
        '''


        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
                'Cookie': 'ABTEST=3|1564648758|v17; IPLOC=JP; SUID=677BC7342113940A000000005D42A536; SUV=1564648795913366; browerV=3; osV=1'
            }
            response = requests.get(url_name, headers=headers, timeout=5)
            self.parse_one_page(response)
            self.company_spidered.add(url_name)
        except Exception as e:
            print(e)

    @funtime
    def run_spider_name(self, name):
        '''
        this function mains to catch the FunctionTimedOut Exception raised by func spider_name()
        :param name:
        :return:
        '''
        try:
            self.spider_name(name)
        except FunctionTimedOut:
            self.dumpToPkl()
            print("dump done")
            sys.exit(0)
        except Exception as e:
            print(e)

    @funtime
    def dumpToPkl(self):
        '''dump the person dict, person_spidered set and person_unspider to the pkl'''

        with open('data/company_spidered.pkl', 'wb') as f:
            pickle.dump(self.company_spidered, f)
        with open('data/company_unspider.pkl', 'wb') as f:
            pickle.dump(self.company_unspider, f)

    @funtime
    def loadFromPkl(self):
        '''

        :return:
        '''
        # the usage of global, which refer to the global variable

        with open('data/company_spidered.pkl', 'rb') as f:
            self.company_spidered = pickle.load(f)
        with open('data/company_unspider.pkl', 'rb') as f:
            self.company_unspider = pickle.load(f)


if __name__ == '__main__':

    count = 0
    a = Spider(False)

    while len(a.company_unspider) > 0:
        time.sleep(random.random() * 3)
        url = a.company_unspider.pop()
        a.run_spider_name(url)
        a.company_spidered.add(url)
        count += 1
        print(count)
        if count % 10 == 0:
            a.dumpToPkl()
            a.loadFromPkl()
        print('\n')