import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import math

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def show(name, img, show=0):
    if show != 0:
        plt.figure(figsize=(6,5)), plt.title(name), plt.get_current_fig_manager()
        plt.imshow(img, cmap='gray'), plt.show(block=False), plt.pause(1), plt.close()


def read(read_img, company_list):
    chars = pytesseract.image_to_string(read_img, lang='eng')
    chars = chars.replace(' ','')
    result_char = ''
    index = 0

    for index in range(len(chars)):
        # put index <= 4 to rid of english carracter in number side  
        # ex) GAOQU601924|[1/e, last e should not be detected.
        if chars[index] == 0 and index <= 5:
            chars[index] == 'O'
        if chars[index].isalpha() and index <= 5:
            result_char += chars[index].upper()
        if chars[index].isdigit():
            result_char += chars[index]
    result_char = cntr_head_adjust(result_char, company_list)
    result_char = cntr_last_digit(result_char)
        
    print("chars : {0}".format(chars) + "result_chars : {0}".format(result_char))
    return chars, result_char
    

########### Cntr Head Adjust - Start ################
def cntr_head_adjust(cntr, company_list):
    ## Adjust Head ##
    if len(cntr) < 5:
        return cntr
    cntr_head = ''
    for _ in range(len(cntr)):
        if cntr[_].isdigit() and _ >= 3:
            break
        cntr_head += cntr[_]
    
    ## If there is no header it should be reversed ##
    if len(cntr_head) == 0:
        return ""

    if cntr_head not in company_list:
        adjusted_head = ''
        if len(cntr_head) < 4:
            adjusted_head = cntr_head_len_under4(cntr_head, company_list)
            cntr = cntr.replace(cntr_head, adjusted_head)
            return cntr
        elif len(cntr_head) == 4:    
            adjusted_head = cntr_head_len_4(cntr_head, company_list)
            cntr = cntr.replace(cntr_head, adjusted_head)
            return cntr
        elif len(cntr_head) > 4:
            adjusted_head = cntr_head_len_over4(cntr_head, company_list)
            cntr = cntr.replace(cntr_head, adjusted_head)
            return cntr

    return cntr


#### Need more detailed code #####
def cntr_head_len_over4(cntr_head, company_list):
    if cntr_head[-1] != 'U':
        return ""
    return cntr_head


def cntr_head_len_under4(cntr_head, company_list):
    if cntr_head[-1] != 'U':
        cntr_head = cntr_head + 'U'
        if len(cntr_head) == 4:
            cntr_head = cntr_head_len_4(cntr_head, company_list)
            return cntr_head

    adjusted_head = ""
    max_accuracy = 0
    for head in company_list:
        accuracy = 0
        for _ in range(len(cntr_head)):
            if cntr_head[_] == head[_]:
                accuracy += 1
            elif cntr_head[_] == head[_+1]:
                accuracy += 1
                _ = _+1
        if accuracy > max_accuracy:
            adjusted_head = head
            max_accuracy = accuracy
    return adjusted_head


def cntr_head_len_4(cntr_head, company_list):
    if cntr_head[-1] != 'U':
        cntr_head[-1] == 'U'
        if cntr_head in company_list:
            return cntr_head
    adjusted_head = ""
    max_accuracy = 0
    for head in company_list:
        accuracy = 0
        for _ in range(3):
            if cntr_head[_] == head[_]:
                accuracy += 1
            if _ == 2 and accuracy == 2 and cntr_head[_] != head[_]:
                if cntr_head[_] == 'O':
                    cntr_head = cntr_head.replace('O','D')
                    accuracy += 1
                elif cntr_head[_] == 'H':
                    cntr_head = cntr_head.replace('H','M')
                    accuracy += 1

        if accuracy > max_accuracy:
            adjusted_head = head
            max_accuracy = accuracy
    return adjusted_head
########### Cntr Head Adjust - End ##################


########### Cntr Last Number - Start ##############
def cntr_last_digit(result_chars):
    Alpha_Weight = [10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38]
    last_digit  = ''
    sum = 0
    if len(result_chars) >= 10:
        for _ in range(0,10):
            if _ == 3: # _ = 3 is always "U"
                sum = sum + 256
            elif _ < 3 and result_chars[_].isalpha() == True:
                sum = sum + Alpha_Weight[int(ord(result_chars[_]))-65] * math.pow(2,_)
            elif _ >= 4 and result_chars[_].isdigit() == True:
                sum = sum + int(result_chars[_]) * math.pow(2,_)
        last_digit = str(int(sum%11))
        if last_digit == '10':
            last_digit = '0'
        # print(last_digit)
        if len(result_chars) == 10:
            result_chars = result_chars + last_digit
        elif len(result_chars) >= 11: ### len(result_chars) == 11 를 len(result_chars) >= 11 로 테스트중
            result_chars = result_chars[0:10]
            result_chars = result_chars + last_digit
    return result_chars
########### Cntr Last Number - End ##############


def img_load(img):
    img_ori = cv2.imread(img)
    img = img.replace("NAV_Noon_Samples\\","").replace(".jpg","")
    return img, img_ori


def img_cut(aim, img_ori):
    w = open(aim)
    cut_range = []
    for _ in range(4):
        cut_range.append(int(w.readline()))

    #### Cut the Img ####
    cutted_img = img_ori[cut_range[0]:cut_range[1], cut_range[2]:cut_range[3]]
    height,width,channel = cutted_img.shape
    # img_denoised = cv2.fastNlMeansDenoisingColored(cutted_img, None, 10,10,41,41)
    return cutted_img,height,width,channel


def gray_black(gray, gr_bl_constant):
    #### Convert Img to Black&White ####
    # (thresh, blackAndWhiteImage) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, gr_bl_constant, 255, cv2.THRESH_BINARY)  # Under the Shadow // 70 or 80 ?
    # (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Out of Shadow
    black = blackAndWhiteImage
    if gr_bl_constant <= 120:
        black = 255 - black
        print('REVERSE')
    return black


def find_contours(height, width, channel, black_copy):
    contours, _ = cv2.findContours(
        black_copy,
        mode = cv2.RETR_TREE,
        # method=cv2.CHAIN_APPROX_SIMPLE
        # method=cv2.CHAIN_APPROX_TC89_KCOS
        method=cv2.CHAIN_APPROX_TC89_L1
        # method=cv2.CHAIN_APPROX_NONE
    )
    temp_result1 = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result1, contours=contours, contourIdx=-1, color=(255,255,255), thickness=1)
    return contours, temp_result1


def prepare_data(height, width, channel, contours):
    temp_result2 = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        ## Delete useless contour
        if h > 40 or h < 20:
            continue
        else:
            cv2.rectangle(temp_result2, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=1)
            contours_dict.append({
                'contour':contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
        })

    return contours_dict, temp_result2


def select_candidates_by_char_size(height, width, channel, contours_dict):
    MIN_AREA = 100
    possible_contours = []
    cnt = 0

    for d in contours_dict:
        area = d['w'] * d['h']
        if MIN_AREA < area:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    
    temp_result3 = np.zeros((height, width, channel), dtype=np.uint8)
    
    for d in possible_contours:
        cv2.rectangle(temp_result3, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)

    return possible_contours, temp_result3


def visualize_possible_cntrs(height, width, channel, matched_result, black_copy):
    temp_result4 = np.zeros((height, width, channel), dtype=np.uint8)

    list_x = []
    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result4, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)
            list_x.append([int(d['x']), int(d['w']), int(d['y']), int(d['h'])])
    list_x.sort()
    # print(list_x)
    # print(len(list_x))

    if len(list_x) == 0:
        return 0, temp_result4, black_copy, list_x

    ## Between character contours and number contours, there is space.
    ## If the space is detected, that space's every color to be black.
    try:
        cv2.rectangle(black_copy, pt1=(list_x[-1][0] + 15 ,0), pt2=(width,height), color=(0,0,0), thickness=-1)
        for index in range(0, len(list_x)-1):
            if list_x[index+1][0] - (list_x[index][0] + list_x[index][1]) >= 20:
                cv2.rectangle(black_copy, pt1=(list_x[index][0]+list_x[index][1]+1, 0), \
                    pt2=(list_x[index+1][0]-1, list_x[index][2]+list_x[index][3]), color=(0,0,0), thickness=-1)
    except:
        return 0, temp_result4, black_copy, list_x
    return 1, temp_result4, black_copy, list_x
#### Select Candidates by Arrangement of Contours ####


def find_chars(contour_list, possible_contours, MAX_DIAG_MULTIPLYER=10, MAX_ANGLE_DIFF=8.0, MIN_N_MATCHED=8):

    matched_result_idx = []
    unmatched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            height_diff = abs(d1['h'] - d2['h'])

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])
        
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)
        unmatched_contour_idx = []

        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        recursive_contour_list = find_chars(unmatched_contour, possible_contours, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MIN_N_MATCHED)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        break
    return matched_result_idx


def rotate_plate_img(height, width, matched_result, black,\
    PLATE_WIDTH_PADDING=3.0, PLATE_HEIGHT_PADDING=1.7, MIN_PLATE_RATIO=3, MAX_PLATE_RATIO=20):

    plate_imgs = []
    plate_infos = []
    show('Before_Rotate', black)

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        ordinary_plate = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x'])
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        side_width = (plate_width - ordinary_plate)/2
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']
            
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
        img_rotated = cv2.warpAffine(black, M=rotation_matrix, dsize=(width, height))
        
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

        ### Plate's Side should be black ###
        img_cropped = cv2.rectangle(img_cropped, pt1=(0,0), pt2=(int(side_width)-5, plate_height), color=(0,0,0), thickness=-1)
        img_cropped = cv2.rectangle(img_cropped, pt1=(int(plate_width) - int(side_width) + 5, 0), pt2=(int(plate_width),int(plate_height)), color=(0,0,0), thickness=-1)
        show("After Rotated", img_cropped)
    try:
        return img_cropped, plate_infos
    except:
        return black, plate_infos


def erosion_detect(img_cropped, company_list, iteration):
    # If without erode, it is clear answer, stop the function
    chars, result_char = read(img_cropped, company_list)

    char = ''.join(filter(str.isalnum, chars))
    # print("TEST : " + char)
    if len(char) >= 6 and char[4].isalpha() == True and char[5].isalpha() == True:
        return "", img_cropped, chars

    if len(result_char) == 11:
        answer = True
        for _ in range(11):
            if _ < 4 and not result_char[_].isalpha():
                answer = False
                break
            if _ > 4 and not result_char[_].isdigit():
                answer = False
                break
        if answer == True:
            return result_char, img_cropped, chars

    if len(chars) == 0:
        return result_char, img_cropped, chars

    img_eroded = img_cropped.copy()

    kernel = np.ones((2,2), np.uint8)
    img_eroded = cv2.erode(img_eroded, kernel, iterations=iteration)

    chars, result_char = read(img_eroded, company_list)

    return result_char, img_eroded, chars


def cntr_size_contour(possible_contours, list_x, height, width, channel):
    x = 0 # First dot for the border line
    y = 0
    z = 0 # Second dot for the border line
    w = 0

    x = list_x[0][0]
    y = list_x[0][2] + list_x[0][3]
    z = list_x[-1][0] + list_x[-1][1]
    w = list_x[-1][2] + list_x[-1][3]

    # y = inclination*x + y_intercept

    inclination = (y-w)/(x-z)  # inclination
    # print(inclination)
    y_intercept = y - inclination*x  # y_intercept
    # y_w_intercept = y_intercept + inclination*width # when x = width, what is y.
    # print(y_w_intercept)

    cntr_size_list = []
    cntr_size_list_x = []
    cntr_size_temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for d in possible_contours:
        tmp = inclination * d['x'] + y_intercept
        if d['y'] + d['h']/2  >= tmp:
            # print("({0} + {1})/2 >= {2}".format(d['y'],d['h'],tmp))
            cntr_size_list.append(d)
            cntr_size_list_x.append([int(d['x']), int(d['w']), int(d['y']), int(d['h'])])
            cv2.rectangle(cntr_size_temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)
    
    show('CNTR_SIZE',cntr_size_temp_result)
    return cntr_size_list, cntr_size_list_x


def cntr_size_contour_draw(cntr_size_result_idx, cntr_size_list_x, black, possible_contours,height,width,channel):
    cntr_size_matched_result = []
    for idx_list in cntr_size_result_idx:
        cntr_size_matched_result.append(np.take(possible_contours, idx_list))

    cntr_size_temp_result = np.zeros((height,width,channel), dtype=np.uint8)

    lowest_y = 0
    highest_h = 0
    for r in cntr_size_matched_result:
        for d in r:
            cv2.rectangle(cntr_size_temp_result,pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)
            if lowest_y < d['y']:
                lowest_y = d['y']
            if d['h'] > highest_h:
                highest_h = d['h']
    cv2.rectangle(black, pt1=(0,0), pt2=(width,lowest_y-5), color=(0,0,0), thickness=-1)
    cv2.rectangle(black, pt1=(0,lowest_y + highest_h + 5), pt2=(width,height), color=(0,0,0), thickness=-1)
    # cv2.rectangle(black, pt1=(0,cntr_size_list_x[0][2]), pt2=(cntr_size_list_x[0][0]-5,height), color=(0,0,0), thickness=-1)
    
    show('CNTR_SIZE_CONTOURS', cntr_size_temp_result)
    return cntr_size_matched_result, black


def main(img, aim, gr_bl_constant, result_possibility,company_list):
    reverse = False
    switch_button = result_possibility # if True, can return // if False, can't return
    ##### IMG LOAD ######
    img2, aim2 = img, aim
    img, img_ori = img_load(img)
    show('img_load', img_ori)

    ##### Range for cut the IMG Load ####
    img_denoised,height,width,channel = img_cut(aim, img_ori)
    show(img + ' cutted_denoised_img', img_denoised)

    #### Img to Grayscale ####
    gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    show(img + ' gray', gray)

    #### Convert Img to Black&White ####
    black = gray_black(gray, gr_bl_constant)
    black_copy = black.copy()
    show(img + ' black', black)
    
    #### Find Contours ####
    contours, temp_result1 = find_contours(height, width, channel, black_copy)
    show(img + ' temp_result1', temp_result1)

    #### Prepare Data ####
    contours_dict, temp_result2 = prepare_data(height, width, channel, contours)
    show(img + ' temp_result2', temp_result2)

    #### Select Candidates by Char Size ####
    possible_contours, temp_result3 = select_candidates_by_char_size(height, width, channel, contours_dict)
    show(img + ' temp_result3', temp_result3)

    result_idx = find_chars(possible_contours, possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    output, temp_result4, black_copy, list_x = visualize_possible_cntrs(height, width, channel, matched_result, black_copy)
    show(img + ' temp_result4', temp_result4)
        
    if output == 0 and switch_button == False:
        gr_bl_constant_reverse = 80
        reverse = True
        # Re-Conduct, black-white-converse
        return main(img2, aim2, gr_bl_constant_reverse, True, company_list)
    elif output == 0 and switch_button == True:
        return "Can't Detect", "Can't Detect", reverse

    #### Rotate Plate Images ####
    img_cropped, plate_infos = rotate_plate_img(height, width, matched_result, black_copy)

    #### Erosion & Detect ####
    iteration = 1
    result_char, img_eroded, chars = erosion_detect(img_cropped, company_list, iteration)
    result_chars = result_char
    show(result_chars, img_eroded)
    
    # Re-Conduct, black-white-converse
    if len(result_chars) < 8 and switch_button == False:
        gr_bl_constant_reverse = 80
        reverse = True
        return main(img2, aim2, gr_bl_constant_reverse, True,company_list)

    #### Result ####
    longest_idx = -1
    try:
        info = plate_infos[longest_idx]
        final_img = img_denoised.copy()
        final_img = cv2.rectangle(final_img, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)
        show(result_chars, final_img)
    except:
        return "Can't Detect", "Can't Detect", reverse

    ### CNTR_SIZE_CHECK ###
    print('CNTR SIZE DETECTING')
    cntr_size_list, cntr_size_list_x = cntr_size_contour(possible_contours, list_x, height, width, channel)
    cntr_size_result_idx = find_chars(cntr_size_list, possible_contours, MAX_DIAG_MULTIPLYER=2, MAX_ANGLE_DIFF=10.0, MIN_N_MATCHED=2)
    cntr_size_matched_result, black_copy = cntr_size_contour_draw(cntr_size_result_idx,cntr_size_list_x, black_copy, possible_contours, height, width, channel)
    cntr_size_img_cropped, cntr_size_plate_infos = rotate_plate_img(height, width, cntr_size_matched_result, black_copy, \
        PLATE_WIDTH_PADDING=3.0, PLATE_HEIGHT_PADDING=3.0, MIN_PLATE_RATIO=3, MAX_PLATE_RATIO=5)
    #### Erosion & Detect ####
    # iteration = 1
    # result_char, cntr_size_img_cropped, chars = erosion_detect(cntr_size_img_cropped, company_list, iteration)
    # result_chars = result_char
    cntr_size_chars, cntr_size_result_char = read(cntr_size_img_cropped, company_list)
    print("CNTR SIZE : ", cntr_size_chars)
    show(cntr_size_chars, cntr_size_img_cropped)

    ### Confirm Container Number ###
    return chars, result_chars, reverse