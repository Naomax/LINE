import MeCab

fi=open("input.txt",'r',encoding="utf-8")
fo=open("output.txt",'r',encoding="utf-8")
fi2=open("input2.txt",'w',encoding="utf-8")
fo2=open("output2.txt",'w',encoding='utf-8')

tagger=MeCab.Tagger("-Owakati")

for row in fi:
    fi2.write(tagger.parse(row))
for row in fo:
    fo2.write(tagger.parse(row))

fi.close()
fo.close()
fi2.close()
fo2.close()