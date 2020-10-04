import MeCab
 
tagger=MeCab.Tagger("-Owakati")
s="こんにちは，ぼくドラえもんです。"

s2_list=[]
test=tagger.parse(s)
print(test)
print(tagger.parse(s)[0],tagger.parse(s)[1],tagger.parse(s)[2])
s2=""
for i in range(len(tagger.parse(s))):
    if (x_test_parse[i]==" ") or (tagger.parse(s)[i]=="　"):
        s2_list.append(s2)
        s2=""
        continue
    print(i)
    s2=s2+tagger.parse(s)[i]
    print(s2)
print(s2_list)