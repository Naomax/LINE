f1=open('input2.txt','r',encoding="utf-8")
f2=open('output2.txt','r',encoding="utf-8")
f3=open('dictionary.txt','w',encoding="utf-8")
W_list=[]
W2_list=[]
for row in f1:
    for i in range(100):
        try:
            W_list.append(row.split()[i])
        except IndexError:
            break
W2_list.append(W_list[0])
for i in range(len(W_list)):#22955
    flag=0
    for j in range(len(W2_list)):
        if W2_list[j] == W_list[i]:
            flag=1
            break
    if flag==1:
        continue
    W2_list.append(W_list[i])
for cnt in range(len(W2_list)):
    print(cnt,W2_list[cnt])
for i in range(len(W2_list)):
    f3.write(W2_list[i]+'\n')
f1.close()
f2.close()
f3.close()            