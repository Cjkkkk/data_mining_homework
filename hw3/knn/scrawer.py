import urllib.request

for i in range(10):
    urllib.request.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", 'data/' + str(i) +".jpg")
    # urllib.request.get("http://jwbinfosys.zju.edu.cn/CheckCode.aspx")
    # contents = urllib.request.urlopen("http://jwbinfosys.zju.edu.cn/CheckCode.aspx").read()
