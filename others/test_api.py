#!/usr/bin/python

import requests
text = "我住在杭州市萧山区明怡花苑95幢101"


li = list()
li.append('我住在杭州市西湖区灵隐寺')
li.append('我在杭州云集这里')
li.append('我叫李大憨，我举报有人在浙江工商大学抢劫')
li.append('你知道三墩镇派出所吗')
li.append('有人在雷峰塔上面吸毒')
li.append('我在华家池这边看到有人在骗人')
li.append('我叫刘少韦华，有个手机是15678374332，电话是58632156')
li.append('我在杭州一知智能科技有限公司上班')
li.append('浙江省杭州市萧山区拱秀路538号')
li.append('明天王二要去杭州学军小学面试')
li.append('我姓刘，要去厦门大学面试')

for text in li:
    print(requests.get("http://192.168.110.8:2603/?s="+text).text)
    print(requests.get("http://192.168.110.8:2604/?s="+text).text)
    print("-----------------------------------------------------")