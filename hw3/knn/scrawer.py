import urllib.request

index = 0
for i in range(5):
    urllib.request.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", str(i) +".jpg")