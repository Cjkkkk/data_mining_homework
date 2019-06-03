import urllib.request


for i in range(100):
    urllib.request.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", 'data/' + str(i) +".jpg")
