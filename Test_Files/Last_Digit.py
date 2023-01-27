import math
import glob

img_files = glob.glob('.\\NAV_Noon_Samples\\*.jpg')

Alpha_Weight = [10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38]
Test_Container = "APHU6566368"
print(Test_Container[0:10])


for img in img_files:
    sum = 0
    img = img.replace(".\\NAV_Noon_Samples\\","").replace(".jpg","")
    print(img)      
    if len(img) >= 10:
        for _ in range(0,10):
            if _ == 3:
                sum = sum + 256
            elif _ < 3 and img[_].isalpha() == True:
                sum = sum + Alpha_Weight[int(ord(img[_]))-65] * math.pow(2,_)
            elif _ >= 4:
                sum = sum + int(img[_]) * math.pow(2,_)
    print(str(int(sum%11)) + " \n")