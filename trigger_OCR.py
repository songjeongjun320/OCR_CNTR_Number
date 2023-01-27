import os
import glob
import Main_by_Trigger
import datetime
import time
import json
import requests
import boto3

# If you change the file location, You should change.
# path_to_watch, img location, img.replace on Main_by_Trigger.py

# original == 150, reverse == 120 // 
# CKOUT == 100, reverse == 50 
gr_bl_constant = 100 # Constant value of threshold
result_possibility = False # To check the condition of

ACCESS_KEY = 'AKIAR76DGZNKHERXNGH4'
SECRET_KEY = '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'


# path -> NGLYOLO/runs/detect
# path_to_watch = C:\Users\ngltr\OneDrive\Desktop\NGL_YOLOV5\runs\detect
# path_to_watch = C:\Users\ngl\Desktop\NGL_YOLOV5\runs\detect
print("\nC:\\Users\\ngltr\\OneDrive\\Desktop\\NGL_YOLOV5\\runs\\detect")
print("C:\\Users\\ngl\\Desktop\\NGL_YOLOV5\\runs\\detect")
print(" --- Find NGLYOLO-runs-detect-way --- " )
print("Path_to_wacth_CCTV : ")
path_to_watch = str(input())

def make_json(result, name, file_path, now, newfolder):  # If right CntrNo detected, send .json file to YMS
    if len(result) == 11 or result != "Can't Detect":
        img_data = name.split('-')
        format = '%m/%d/%Y %H:%M:%S'
        d = dict()
        d['division'] = img_data[0]
        d['cameraPosition'] = img_data[1]
        d['cntrNo'] = result
        d['imgPath'] = newfolder + "/" + name + ".jpg"
        d['imgNm'] = name+".jpg"
        d['detectedTime'] = now.strftime(format)

        f_name = name
        json_path = file_path + '\\' + f_name + ".json"
        with open (json_path, 'w', encoding='UTF-8') as fp:
            json.dump(d, fp, indent=4)
        return True  # If right CntrNo
        # print("File_name ", f_name)
    return False  # If wrong CntrNo


def api_to_yms(fp):
    url = "http://35.85.252.213:8080/api/v1/ocr"
    with open(fp, encoding='UTF-8') as json_file:
        data = json.load(json_file)
    headers = {'Content-type': 'application/json', 'Accept': '*/*', 'Cookie' : 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIxIiwiaWF0IjoxNjY3NDI2NDA5LCJleHAiOjE2Njc1MTI4MDksInJvbGUiOiJST0xFX1VTRVIifQ.64v_dKMvak6pSjLpKCL1imDO6ma4oyHWJLsG_AFbC5guR-YoR6AB7gb9CJi-cx-fu1jK13U3tvwkOhpY8WQG4g'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    print("Status code: ", r.status_code)
    return r.status_code, data["imgPath"]  # "010620234/CKOUT-153529-15355221.jpg"


def img_to_yms(img_path, fn):  # Img send to YMS, ex fn = 01052023
    img_name = img_path.split('\\')
    print("fn before : ", fn)
    fn = fn.split('/')
    fn = fn[0]
    # print(img_name)
    img_name = img_name[-1]
    print("fn : ", fn)
    print("img_name :", img_name)
    s3 = boto3.client('s3',
                        region_name='us-west-2',
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY)
    s3.upload_file(img_path, 'ngl-yms', fn + "/" + img_name)


def crops(files, dir, newfolder, company_list):
    now = datetime.datetime.now()
    dateformat = '%m%d%Y'
    success_detected = False
    for file in files:
        if str(file) == "crops":
            dir = dir + "\\crops\\Container_Number"
            img_files = glob.glob(dir + "\\*.jpg")  #   print("img_files ", img_files)
            
            for n in range(9999):  # Create folder for json files
                file_path = '.\\JSON_API\\'+now.strftime(dateformat)  # .\JSON_API\01012023
                if n != 0 : file_path = file_path + str(n)  # .\JSON_API\010120231
                if os.path.isdir(file_path) == False:
                    os.mkdir(file_path)
                    break

            for img in img_files:  # Reading the images
                if os.path.isdir(img) == False:
                    name,ext = os.path.splitext(img)
                    name = os.path.basename(name)
                    print("jpg_file_name : ", name)
                    camera_position = name.split('-')[1]
                    print(camera_position)

                    if camera_position == "CKOUT":
                        gr_bl_constant = 100
                        gr_bl_constant_reverse = 50
                        h_max = 40
                        h_min = 20

                    elif camera_position == "CKIN":
                        gr_bl_constant = 150
                        gr_bl_constant_reverse = 100
                        h_max = 150
                        h_min = 50

                    if str(ext) == ".jpg":
                        print("\n")
                        chars, result, reverse = Main_by_Trigger.main(img, gr_bl_constant, gr_bl_constant_reverse, h_max, h_min, result_possibility, company_list)                    
                        print("Results-- : " + result)
                        if make_json(result, name, file_path, now, newfolder): success_detected = True

    if success_detected == True and len(glob.glob(file_path + "\\*.json")) != 0:  # If it is not empty, send API to YMS
        file = glob.glob(file_path + "\\*.json")
        fp = file[0]
        print("json_file_name : ", file[0])
        status_code, img_path = api_to_yms(fp)

        if status_code == 200:  # 200 means success API to YMS
            fn = img_path.split('\\')
            img_path = img_path.replace("/", "\\")
            print("IMG_PATH : ", img_path)
            print("Status_Code : " , fn)
            img_path = path_to_watch + "\\" + img_path.replace("\\","\\crops\\Container_Number\\")
            print("img_path : ", img_path)  # 010620234\crops\Container_Number\CKOUT-153529-15355221.jpg,
            img_to_yms(img_path, fn[0])

def company_list_download():
    # Company_List DB create to adjust the Character
    c = open("Company_List\Company_List.txt", 'r')
    lines = c.readlines()
    company_list = []

    ########## When the container list can be extracted by TMS, company_list --> container_list #########

    for line in lines:
        line = line.replace('\n','')
        company_list.append(line)
    c.close()
    return company_list

def run():
    old = os.listdir(path_to_watch)
    company_list = company_list_download()
    print("READY TO RUN - OCR_Engine")
    while True:
        new = os.listdir(path_to_watch)
        if len(new) > len(old):
            newfolder = list(set(new) - set(old))
            print("New_File : ", newfolder)
            for n in range(len(newfolder)):
                print("Reading_File : ", newfolder[n])
                print("Reading Start : ")
                time.sleep(60)
                # print("new ", len(new))
                # print("old ", len(old))
                old = new
                extension = os.path.splitext(path_to_watch + "\\" + newfolder[n])[1]
                if extension == "":
                    dir = path_to_watch + "\\" + newfolder[n]  # path_to_watch\01052023
                    # print("dir ", dir)
                    files = os.listdir(dir)
                    # print("files ", files)
                    crops(files, dir, newfolder[n], company_list)
                else:
                    continue
                print("Readed\n")
        else:
            continue


run()