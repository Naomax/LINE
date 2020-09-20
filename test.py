import MeCab
 
tagger=MeCab.Tagger("-owakati")

print(tagger.parse("こんにちは。俺の名前はルフィ。海賊王になる男だ。"))