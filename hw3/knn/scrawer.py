import urllib.request


# 爬取图片并存在data目录下，文件格式为i.jpg， i为[0..100]
for i in range(100):
    urllib.request.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", 'data/' + str(i) +".jpg")
