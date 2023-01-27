import glob
import Test_NAV
import Test_NAV_Header

# img_files = glob.glob('.\\Samples_HQ\\*.jpg')
# range_files = glob.glob('.\\Samples_HQ\\*.txt')
# img_files = glob.glob('.\\Samples\\*.jpg')
# range_files = glob.glob('.\\Samples\\*.txt')
img_files = glob.glob('.\\NAV_Noon_Samples\\*.jpg')
range_files = glob.glob('.\\NAV_Noon_Samples\\*.txt')

i = 0
true = 0
false = 0
head_issue = 0
gr_bl_constant = 150 # Constant value of threshold
result_possibility = False # To check the condition of

a = open("result_success.txt", 'w', encoding='UTF-8')
k = open("result_fail.txt", 'w', encoding='UTF-8')
h = open("result_header_issue.txt", 'w', encoding='UTF-8')

# Company_List DB create to adjust the Character
c = open("Company_List\Company_List.txt", 'r')
lines = c.readlines()
company_list = []
for line in lines:
    line = line.replace('\n','')
    company_list.append(line)
c.close()

for img in img_files:
    former = ''
    aim = range_files[i]
    i = i+1
    aim = aim.replace(".\\",'')
    img = img.replace(".\\",'')
    answer = img
    name = answer

    name = name.replace("NAV_Noon_Samples\\","").replace(".jpg","")
    print("\n" + name)
    
    cntr_number, result, reverse, cntr_size = Test_NAV.main(img, aim, gr_bl_constant, result_possibility, company_list)
    # cntr_number, result, reverse = Test_NAV_Header.main(img, aim)
    # cntr_number, result, reverse = Test.main(img, aim)

    # answer = answer.replace("Samples_HQ\\","")[0:11]
    answer = answer.replace("NAV_Noon_Samples\\","")[0:11]
    # answer = answer.replace("Samples\\","")[0:11]

    if answer == result:
        true = true+1
        a.write("================================\n")
        a.write("Answer : " + name + "\n")
        a.write("Cntr_number  : " + cntr_number)
        a.write("Cntr_Size : " + cntr_size, "\n")
        a.write("Result : " + result + "\n")
        a.write("Success Rate : {0}".format(true/(true+false)*100)+ "\n")
    else:
        false = false+1
        if former != name[0:11]:
            k.write("\n")
        k.write("================================\n")
        if cntr_number == "Can't Detect":
            k.write("Can't Detect")
        if name[0:4] != result[0:4]:
            k.write("Header Issue \n")
            h.write("================================\n")
            h.write("Answer : " + name + "\n")
            h.write("Cntr_number  : " + cntr_number)
            h.write("Cntr_Size : " + cntr_size, "\n")
            h.write("Result : " + result + "\n")
            h.write("Success Rate : {0}".format(true/(true+false)*100)+ "\n")
            if reverse == True:
                h.write("reverse\n")
            head_issue = head_issue + 1
        if reverse == True:
            k.write("reverse\n")
        k.write("Answer : " + name + "\n")
        k.write("Cntr_number  : " + cntr_number)
        k.write("Cntr_Size : " + cntr_size, "\n")
        k.write("Result : " + result + "\n")
        k.write("Success Rate : {0}".format(true/(true+false)*100)+ "\n")
    former = name[0:11]
    print("Success Rate : {0}".format(true/(true+false)*100) + "\n")


h.write("\n================================\n")
h.write("\n================================\n")
h.write("Total : " + str(i) + "\n")
h.write("Failure : " + str(false) + "\n")
h.write("Head Issue : "+ str(head_issue) + "\n")
h.write("================================\n")
h.close()

k.write("\n================================\n")
k.write("\n================================\n")
k.write("Total : " + str(i) + "\n")
k.write("Success : " + str(true) + "\n")
k.write("Failure : " + str(false) + "\n")
k.write("Head Issue : "+ str(head_issue) + "\n")
k.write("Success Rate : " + str(true/(true+false)*100) + "\n")
k.write("================================\n")
k.close()

a.write("\n================================\n")
a.write("\n================================\n")
a.write("Total : " + str(i) + "\n")
a.write("Success : " + str(true) + "\n")
a.write("Failure : " + str(false) + "\n")
a.write("Head Issue : "+ str(head_issue) + "\n")
a.write("Success Rate : " + str(true/(true+false)*100) + "\n")
a.write("================================\n")
a.close()