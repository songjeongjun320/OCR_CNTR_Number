import glob

img_files = glob.glob('.\\NAV_Noon_Samples\\*.jpg')

k = open("Company_List.txt", 'w')
company_list = []

for img in img_files:
    img = img.replace(".\\NAV_Noon_Samples\\","").replace(".jpg","")
    img = img[0:4]
    if img not in company_list:
        company_list.append(img)
        k.write(img+"\n")
        print(img)
k.close()