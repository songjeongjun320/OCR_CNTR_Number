import glob
import Main

img_files = glob.glob('.\\Test_Img\\*.jpg')

i = 0
true = 0
false = 0
head_issue = 0
gr_bl_constant = 150 # Constant value of threshold
result_possibility = False # To check the condition of

a = open("result_success.txt", 'w')
k = open("result_fail.txt", 'w')

# Company_List DB create to adjust the Character
c = open("Company_List.txt", 'r')
lines = c.readlines()
company_list = []
for line in lines:
    line = line.replace('\n','')
    company_list.append(line)
c.close()

for img in img_files:
    former = ''
    i = i+1
    img = img.replace(".\\",'')
    answer = img
    name = answer

    name = name.replace("Test_Img\\","").replace(".jpg","")
    print("\n" + name)
    
    chars, result, reverse = Main.main(img, gr_bl_constant, result_possibility, company_list)

    answer = answer.replace("Test_Img\\","")[0:11]

    if answer == result:
        true = true+1
        a.write("================================\n")
        a.write("Answer : " + name + "\n")
        a.write("Chars  : " + chars)
        a.write("Result : " + result + "\n")
        a.write("Success Rate : {0}".format(true/(true+false)*100)+ "\n")
    else:
        false = false+1
        if former != name[0:11]:
            k.write("\n")
        k.write("================================\n")
        if chars == "Can't Detect":
            k.write("Can't Detect")
            if reverse == True:
                h.write("reverse\n")
            head_issue = head_issue + 1
        if reverse == True:
            k.write("reverse\n")
        k.write("Answer : " + name + "\n")
        k.write("Chars  : " + chars)
        k.write("Result : " + result + "\n")
        k.write("Success Rate : {0}".format(true/(true+false)*100)+ "\n")
    former = name[0:11]
    print("Success Rate : {0}".format(true/(true+false)*100) + "\n")

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