"""
f=open("dictionary.txt","r",encoding="utf-8")
W_list=[]
W2_list=[]
W3_list=[]
for row in f:
    #print(row)
    W_list.append(row)
print(W_list)
f.close()
f=open("dictionary.txt","a",encoding="utf-8")
fo=open("output2.txt","r",encoding="utf-8")
for row in fo:
    for i in range(100):
        try:
            W2_list.append(row.split()[i]+"\n")
        except IndexError:
            break
for i in range(len(W2_list)):
    print(i,len(W2_list),len(W_list))
    flag=0
    for j in range(len(W_list)):
        if W_list[j]==W2_list[i]:
            flag=1
            break
    if flag==1:
        continue
    W_list.append(W2_list[i])
    W3_list.append(W2_list[i])
for i in range(len(W3_list)):
    f.write(W3_list[i])
fo.close()
f.close()
"""
f=open("dictionary.txt","r",encoding="utf-8")
cnt=0
for row in f:
    print(cnt,row)
    cnt=cnt+1
f.close()