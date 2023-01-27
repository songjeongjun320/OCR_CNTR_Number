# c = open("Company_List.txt", 'r')
# lines = c.readlines()
# company_list = []
# for line in lines:
#     line = line.replace('\n','')
#     company_list.append(line)
# print(company_list)


cntr_head = "AHU"
c = open("Company_List.txt", 'r')
lines = c.readlines()
company_list = []
for line in lines:
    line = line.replace('\n','')
    company_list.append(line)
c.close()

def cntr_head_len_over_4(cntr_head, company_list):
    adjusted_head = ''
    for _ in range(0, len(cntr_head)):
        head = cntr_head.replace(cntr_head[_],'')
        for company in company_list:
            if head in company:
                adjusted_head = company
                return adjusted_head

result_chars = cntr_head_len_over_4(cntr_head, company_list)

print(result_chars)

"""
YMMU
YMLU
TCNU
TCKU
CAAU
CAIU
"""